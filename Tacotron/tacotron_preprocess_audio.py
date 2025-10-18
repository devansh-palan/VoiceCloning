from preprocess import preprocess_dataset
from pathlib import Path

datasets_root = Path(r"LibriSpeech/train-clean-100")


out_dir = Path(r"LibriSpeech") / "processed" / "synthesizer"
datasets_name = "LibriSpeech"
out_dir.mkdir(exist_ok=True, parents=True)


preprocess_dataset(
    datasets_root=datasets_root,
    out_dir=out_dir,
    datasets_name=datasets_name,
   
)
