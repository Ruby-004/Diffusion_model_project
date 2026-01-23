import os
import os.path as osp
import random
from urllib.parse import urlparse
import requests
import zipfile

from tqdm import tqdm


# Zenodo URLs for dataset and pre-trained models
# All files are hosted at: https://zenodo.org/records/18341260
ZENODO_RECORD_ID = "18341260"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"

# Download URLs
DATASET_URL = f"{ZENODO_BASE_URL}/dataset_3d.zip?download=1"
VAE_MODELS_URL = f"{ZENODO_BASE_URL}/VAE%27s.zip?download=1"
DIFFUSION_MODEL_URL = f"{ZENODO_BASE_URL}/20260120_unet_latent-diffusion_in-17-out-8-f-5-k-3-p-zeros-a-3..2-dr-0.0-wd-0.00e%2B00-b-2-lr-1.00e-03-ep-104.zip?download=1"

# DOI for citation
ZENODO_DOI = "10.5281/zenodo.18341260"


user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0"
]

def download_data(url: str, save_dir: str) -> str:
    """
    Download data from URL.

    Args:
        url: URL of the data.
        save_dir: directory where data is stored.
    """

    if not osp.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    zip_filename = get_zip_filename(url)
    zip_path = osp.join(save_dir, zip_filename)

    if osp.exists(zip_path):
        print(f'File "{zip_path}" already exists. Skipping download.')

    else:
        print(f'Downloading data from "{url}" ...')
        headers = {"User-Agent": random.choice(user_agents)} # add header to avoid being detected as bot
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with tqdm(total=total_size, unit='B', unit_scale=True, desc=zip_filename) as pbar:
            chunk_size = 1024 * 1024  # 1 MB
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f'Data downloaded to "{zip_path}".')

    return zip_path


def unzip_data(zip_path: str, save_dir: str) -> str:
    """
    Extract data from zip file.

    Args:
        zip_path: path to the zip file.
        save_dir: directory where data is extracted to.
    """
    print(f'Extracting data from "{zip_path}" ...')

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # the zip file contains a single folder (with subfolders/files)
        namelist = zip_ref.namelist()
        folder_name = namelist[0].split('/')[0]

        folder_path = osp.join(save_dir, folder_name)
        if osp.exists(folder_path):
            print(f'Folder "{folder_path}" already exists. Skipping extraction.')
        else:
            print(f'Extracting data from "{zip_path}" ...')
            zip_ref.extractall(save_dir)
            print(f'Data extracted to "{folder_path}".')
    
    return folder_path


def is_url(s: str) -> bool:
    """Return True if string is a valid URL (http or https)."""
    try:
        parsed = urlparse(s.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def get_zip_filename(link: str) -> str:
    """
    Get the zip filename from a zenodo download link.

    Args:
        link: zenodo download link that is like 'https://zenodo.org/records/18341260/files/dataset_3d.zip?download=1'
    
    Returns:
        out: zip filename, i.e., 'dataset_3d.zip'.
    """
    out = link.split('/')[-1].split('?')[0]
    return out


def download_dataset(save_dir: str = "./dataset_3d") -> str:
    """
    Download the dataset from Zenodo and extract it.
    
    Args:
        save_dir: Directory where dataset will be extracted to.
        
    Returns:
        Path to the extracted dataset folder.
    """
    parent_dir = osp.dirname(osp.abspath(save_dir))
    zip_path = download_data(DATASET_URL, parent_dir)
    folder_path = unzip_data(zip_path, parent_dir)
    os.remove(zip_path)
    print(f'Removed zip file: "{zip_path}"')
    return folder_path


def download_vae_models(save_dir: str = "./VAE_model/trained") -> str:
    """
    Download the pre-trained VAE models from Zenodo and extract them.
    
    Args:
        save_dir: Directory where VAE models will be extracted to.
        
    Returns:
        Path to the extracted folder.
    """
    os.makedirs(save_dir, exist_ok=True)
    zip_path = download_data(VAE_MODELS_URL, save_dir)
    folder_path = unzip_data(zip_path, save_dir)
    os.remove(zip_path)
    print(f'Removed zip file: "{zip_path}"')
    return folder_path


def download_diffusion_model(save_dir: str = "./Diffusion_model/trained") -> str:
    """
    Download the pre-trained diffusion model from Zenodo and extract it.
    
    Args:
        save_dir: Directory where diffusion model will be extracted to.
        
    Returns:
        Path to the extracted folder.
    """
    os.makedirs(save_dir, exist_ok=True)
    zip_path = download_data(DIFFUSION_MODEL_URL, save_dir)
    folder_path = unzip_data(zip_path, save_dir)
    os.remove(zip_path)
    print(f'Removed zip file: "{zip_path}"')
    return folder_path