
from huggingface_hub import HfApi

repo_id = "maha16694/tourism-package"

api = HfApi()

api.upload_file(
    path_or_fileobj="tourism_project/data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id=repo_id,
    repo_type="dataset" 
)

print("Dataset uploaded successfully to Hugging Face")
