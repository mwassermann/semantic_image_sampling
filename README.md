This is code to train and test an index based on the datacomp 12M dataset. The trained index is hosted on huggingface and can be loaded with the function below.




from huggingface_hub import hf_hub_download

for filename in ["index.faiss", "faiss_uids.npy", "metadata.parquet"]:
    hf_hub_download(
        repo_id="your-username/datacomp-12m-index",
        repo_type="dataset",
        filename=filename,
        local_dir="./datacomp",
    )
