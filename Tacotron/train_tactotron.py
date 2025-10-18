from pathlib import Path
from train import train



if __name__ == "__main__":

    run_id = "tacotron_training"
    syn_dir = Path(r"LibriSpeech\processed\synthesizer")  
    models_dir = Path(r"Tacotron\utils\saved_model")
    save_every = 1000
    backup_every = 25000
    force_restart = False
    

    args = {
        "run_id": run_id,
        "syn_dir": syn_dir,
        "models_dir": models_dir,
        "force_restart": force_restart
    }

    train(**args)