"""
setup_index.py — DataComp-12M FAISS index builder
Sequential processing, UID mapping, checkpointing, detailed logging.
"""

import os
import io
import json
import time
import tarfile
import logging
import numpy as np
import faiss
import pandas as pd
from datetime import datetime
from huggingface_hub import hf_hub_download

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR       = "./datacomp"
TMP_DIR        = os.path.join(SAVE_DIR, "emb_tmp")
INDEX_PATH     = os.path.join(SAVE_DIR, "index.faiss")
PARTIAL_PATH   = os.path.join(SAVE_DIR, "index.faiss.partial")
UID_MAP_PATH   = os.path.join(SAVE_DIR, "faiss_uids.npy")
UID_MAP_PARTIAL= os.path.join(SAVE_DIR, "faiss_uids.partial.npy")
PROGRESS_PATH  = os.path.join(SAVE_DIR, "index_progress.json")
LOG_PATH       = os.path.join(SAVE_DIR, "build.log")

DIM            = 1536
N_TRAIN_SHARDS = 13      # 13 × ~12,500 = ~163K vectors for IVF training
NLIST          = 4096
M_PQ           = 48
NBITS          = 8
NPROBE         = 64
CHECKPOINT_INT = 50      # save partial index every N shards
HF_REPO        = "apple/DataCompDR-12M"
HF_TOKEN       = "hf_AARWnUOVPwvuwngczRHoKNLdWHJPMG"      # set your token here

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


def log_disk():
    """Log current disk usage of key files and free space."""
    stat = os.statvfs(SAVE_DIR)
    free_gb = stat.f_bavail * stat.f_frsize / 1e9
    sizes = {}
    for name, path in [
        ("index.faiss", INDEX_PATH),
        ("index.partial", PARTIAL_PATH),
        ("faiss_uids", UID_MAP_PATH),
        ("faiss_uids.partial", UID_MAP_PARTIAL),
    ]:
        if os.path.exists(path):
            sizes[name] = f"{os.path.getsize(path)/1e9:.2f}GB"
    log.info(f"Disk free: {free_gb:.1f}GB | Files: {sizes}")


def get_shard_list():
    """Return ordered list of shard IDs from manifest."""
    manifest_path = os.path.join(TMP_DIR, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        log.info("Downloading manifest...")
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename="manifest.jsonl",
            local_dir=TMP_DIR,
            token=HF_TOKEN or None,
        )
    shards = []
    with open(manifest_path) as f:
        for line in f:
            shards.append(json.loads(line)["shard"])
    shards.sort()
    log.info(f"Found {len(shards)} shards in manifest")
    return shards


def download_shard(shard_id):
    """Download a shard tar, return local path. Skips if already present."""
    local_path = os.path.join(TMP_DIR, f"{shard_id}.tar")
    if os.path.exists(local_path):
        log.info(f"  {shard_id}: already on disk, skipping download")
        return local_path
    t0 = time.time()
    hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=f"{shard_id}.tar",
        local_dir=TMP_DIR,
        token=HF_TOKEN or None,
    )
    elapsed = time.time() - t0
    size_mb = os.path.getsize(local_path) / 1e6
    log.info(f"  {shard_id}: downloaded {size_mb:.0f}MB in {elapsed:.1f}s "
             f"({size_mb/elapsed:.1f} MB/s)")
    return local_path


def extract_shard(tar_path):
    """
    Extract embeddings and UIDs from a tar shard.
    Returns:
        vecs: np.ndarray (N, 1536) float32, each row normalized
        uids: list of N uid strings
    """
    vecs, uids = [], []
    with tarfile.open(tar_path) as tar:
        members = sorted(
            [m for m in tar.getmembers() if m.name.endswith(".npz")],
            key=lambda m: m.name
        )
        for member in members:
            uid = os.path.splitext(os.path.basename(member.name))[0]
            raw = tar.extractfile(member).read()
            data = np.load(io.BytesIO(raw))
            # image_emb shape: (30, 1536) — 30 augmentations
            vec = data["image_emb"].mean(axis=0).astype(np.float32)
            vecs.append(vec)
            uids.append(uid)
    return np.stack(vecs), uids


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            p = json.load(f)
        log.info(f"Resuming from checkpoint: {len(p['completed'])} shards done, "
                 f"trained={p['trained']}")
        return p
    return {"completed": [], "trained": False}


def save_progress(progress):
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f)


def build_index():
    log.info("=" * 70)
    log.info("DataComp-12M index builder starting")
    log.info(f"DIM={DIM}  nlist={NLIST}  M={M_PQ}  nbits={NBITS}")
    log.info("=" * 70)
    log_disk()

    if os.path.exists(INDEX_PATH) and not os.path.exists(PROGRESS_PATH):
        log.info("Final index already exists and no progress file found — done.")
        return

    shards = get_shard_list()
    progress = load_progress()
    completed = set(progress["completed"])

    # ── Build or restore index ────────────────────────────────────────────────
    if progress["trained"] and os.path.exists(PARTIAL_PATH):
        log.info(f"Loading partial index from checkpoint ({PARTIAL_PATH})...")
        index = faiss.read_index(PARTIAL_PATH)
        index.nprobe = NPROBE
        uid_list = list(np.load(UID_MAP_PARTIAL, allow_pickle=True))
        log.info(f"Restored index: {index.ntotal:,} vectors, "
                 f"{len(uid_list)} UIDs in map")
        assert index.ntotal == len(uid_list), (
            f"Vector/UID count mismatch on resume: "
            f"{index.ntotal} vs {len(uid_list)}"
        )
    else:
        quantizer = faiss.IndexFlatIP(DIM)
        index = faiss.IndexIVFPQ(
            quantizer, DIM, NLIST, M_PQ, NBITS, faiss.METRIC_INNER_PRODUCT
        )
        index.nprobe = NPROBE
        uid_list = []

    total_added = index.ntotal

    # ── Phase 1: train ────────────────────────────────────────────────────────
    if not progress["trained"]:
        log.info(f"Phase 1: training on {N_TRAIN_SHARDS} shards sequentially")
        train_vecs_list = []
        train_uid_lists = []

        for i, shard_id in enumerate(shards[:N_TRAIN_SHARDS]):
            log.info(f"  Training shard [{i+1}/{N_TRAIN_SHARDS}] {shard_id}")
            local_path = download_shard(shard_id)
            vecs, uids = extract_shard(local_path)
            os.remove(local_path)
            log.info(f"    Extracted {len(vecs)} vectors, deleted tar")
            train_vecs_list.append(vecs)
            train_uid_lists.append(uids)

        train_vecs = np.concatenate(train_vecs_list, axis=0)
        log.info(f"Training on {len(train_vecs):,} vectors "
                 f"(need >{39 * NLIST:,})...")
        assert len(train_vecs) >= 39 * NLIST, (
            f"Not enough training vectors: {len(train_vecs)} < {39*NLIST}"
        )
        faiss.normalize_L2(train_vecs)
        t0 = time.time()
        index.train(train_vecs)
        log.info(f"Training done in {time.time()-t0:.1f}s, "
                 f"is_trained={index.is_trained}")

        # Add training vectors immediately — no re-download needed
        log.info("Adding training shard vectors to index...")
        for shard_id, vecs, uids in zip(
            shards[:N_TRAIN_SHARDS], train_vecs_list, train_uid_lists
        ):
            faiss.normalize_L2(vecs)
            index.add(vecs)
            uid_list.extend(uids)
            total_added += len(vecs)
            progress["completed"].append(shard_id)
            completed.add(shard_id)
            log.info(f"  Added {shard_id}: {len(vecs)} vecs, "
                     f"{total_added:,} total")

        del train_vecs, train_vecs_list, train_uid_lists
        progress["trained"] = True
        save_progress(progress)
        np.save(UID_MAP_PARTIAL, np.array(uid_list))
        faiss.write_index(index, PARTIAL_PATH)
        log.info(f"Phase 1 complete. Checkpoint saved. {total_added:,} vectors.")
        log_disk()

    # ── Phase 2: remaining shards ─────────────────────────────────────────────
    remaining = [s for s in shards if s not in completed]
    log.info(f"Phase 2: {len(remaining)} shards remaining (sequential)")

    for i, shard_id in enumerate(remaining):
        global_i = shards.index(shard_id)
        t0 = time.time()
        log.info(f"[{global_i+1}/{len(shards)}] Processing {shard_id} "
                 f"({len(remaining)-i} remaining)...")

        local_path = download_shard(shard_id)
        vecs, uids = extract_shard(local_path)
        os.remove(local_path)
        log.info(f"  Extracted {len(vecs)} vecs, tar deleted")

        faiss.normalize_L2(vecs)
        index.add(vecs)
        uid_list.extend(uids)
        total_added += len(vecs)

        assert index.ntotal == len(uid_list), (
            f"MISMATCH after shard {shard_id}: "
            f"index={index.ntotal} uids={len(uid_list)}"
        )

        progress["completed"].append(shard_id)
        save_progress(progress)

        elapsed = time.time() - t0
        log.info(f"  Done in {elapsed:.1f}s — "
                 f"{total_added:,} vectors total, "
                 f"{len(uid_list):,} UIDs tracked")

        if (global_i + 1) % CHECKPOINT_INT == 0:
            log.info(f"  Saving checkpoint...")
            faiss.write_index(index, PARTIAL_PATH)
            np.save(UID_MAP_PARTIAL, np.array(uid_list))
            log.info(f"  Checkpoint saved.")
            log_disk()

    # ── Finalise ──────────────────────────────────────────────────────────────
    log.info("Saving final index and UID map...")
    faiss.write_index(index, INDEX_PATH)
    np.save(UID_MAP_PATH, np.array(uid_list))

    for path in [PARTIAL_PATH, UID_MAP_PARTIAL, PROGRESS_PATH]:
        if os.path.exists(path):
            os.remove(path)
            log.info(f"Removed {path}")

    log.info("=" * 70)
    log.info(f"Build complete!")
    log.info(f"  Vectors in index : {index.ntotal:,}")
    log.info(f"  UIDs in map      : {len(uid_list):,}")
    log.info(f"  Index saved to   : {INDEX_PATH}")
    log.info(f"  UID map saved to : {UID_MAP_PATH}")
    log.info("=" * 70)
    log_disk()


if __name__ == "__main__":
    build_index()