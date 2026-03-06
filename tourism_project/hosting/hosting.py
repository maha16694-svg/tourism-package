from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="tourism_project/deployment",   # folder containing Dockerfile, app.py, requirements.txt
    repo_id="maha16694-svg/tourism-package",    # your Hugging Face Space repo
    repo_type="space",
    path_in_repo=""
)
