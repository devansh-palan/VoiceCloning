from pathlib import Path
from .train import train

def train_vocoder_model(run_id: str,datasets_root: Path,syn_dir: Path = None,voc_dir: Path = None,models_dir: Path = Path("saved_models"),
    force_restart: bool = False
):
    
    if syn_dir is None:
        syn_dir = datasets_root / "processed" / "synthesizer"
    if voc_dir is None:
        voc_dir = datasets_root / "processed" / "vocoder"

    models_dir.mkdir(exist_ok=True)

    args = {
        "run_id": run_id,
        "syn_dir": syn_dir,
        "voc_dir": voc_dir,
        "models_dir": models_dir,
        "force_restart": force_restart,
    }

    train(**args)

if __name__ == "__main__":
    train_vocoder_model(
        run_id="vocoder_run_01",
        datasets_root=Path(r"LibriSpeech"),     
        force_restart=False,     
    )
