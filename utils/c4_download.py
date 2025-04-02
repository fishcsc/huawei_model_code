import os
os.environ['HF_HOME']='/data/hfhub'
os.environ['HF_ENDPOINT']='https://hf-mirror.com'

from huggingface_hub import snapshot_download

# Download allenai/c4 en dataset
snapshot_download(
    repo_id="allenai/c4",
    repo_type="dataset",
    allow_patterns='en/*',
    resume_download=True
)