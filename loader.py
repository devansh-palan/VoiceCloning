import os
import gdown

# Folder to store the models
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Model file names and corresponding Google Drive file IDs
MODEL_FILES = {
    'encoder.pt':     '1NV3ooEwd_XHxUIgb0VnBYHmcRJJZkMXV',
    'synthesizer.pt': '1yVOxEyMQsnfu4bRxdoKeEKvCt332TsD-',
    'vocoder.pt':     '1cF3v0Tj3PpXgSTjPlQ7gpTVKXq8SrZyM'
}

def download_model(file_name, file_id):
    file_path = os.path.join(MODEL_DIR, file_name)
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        try:
            gdown.download(url, file_path, quiet=False)
            print(f"Downloaded {file_name} to {file_path}")
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")
            return None
    else:
        print(f"{file_name} already exists.")
    
    return file_path

def load_voice_cloning_models():
    paths = {}
    for name, fid in MODEL_FILES.items():
        path = download_model(name, fid)
        if path:
            paths[name] = path
    return paths
