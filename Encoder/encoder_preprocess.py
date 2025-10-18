from preprocess import preprocess_librispeech
from pathlib import Path

datasets_root = Path(r"../LibriSpeech")   
out_dir = datasets_root/"processed"/ "encoder"


out_dir.mkdir(parents=True, exist_ok=True)


preprocess_librispeech(
    dataset_root=datasets_root,
    out_dir=out_dir,
    skip_existing=False  
)
