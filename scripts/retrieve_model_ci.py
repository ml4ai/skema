import os
from pathlib import Path

from huggingface_hub import hf_hub_download

def retrieve_model(model_path=None) -> str:
    """
    Retrieve the img2mml model from the specified path or download it if not found.

    Args:
        model_path (str, optional): Path to the img2mml model file. Defaults to None.

    Returns:
        str: Path to the loaded model file.
    """
    cwd = Path(__file__).parents[0]
    REPO_NAME = "lum-ai/img2mml"
    MODEL_NAME = "cnn_xfmer_arxiv_im2mml_with_fonts_boldface_best.pt"
    # If the model path is none or doesn't exist, the default model will be downloaded from server.
    if model_path is None or not os.path.exists(model_path):
        model_path = cwd / "trained_models" / MODEL_NAME

        # Check if the model file already exists
        if not os.path.exists(model_path):
            # If the file doesn't exist, download it from the specified URL
            print(f"Downloading the model checkpoint from HuggingFace...")
            hf_hub_download(repo_id=REPO_NAME, filename=MODEL_NAME, local_dir=model_path.parent, local_dir_use_symlinks=False)
        
    return str(model_path)

retrieve_model()