from pathlib import Path
import torch
from tacotron_dataset import SynthesizerDataset,collate_synthesizer
from model import Tacotron
from utils.text import symbols
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from functools import partial
def generate_gta_mels(datasets_root=Path(r"LibriSpeech"), syn_model_fpath=Path(r"Tacotron/utils/saved_model/tacotron_training/synthesizer.pt")):
    
    in_dir = datasets_root / "processed" / "synthesizer"
    out_dir = datasets_root / "processed" / "vocoder"
    synth_dir = out_dir / "mels_gta"
    synth_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Tacotron(
        embed_dims=512,
        num_chars=len(symbols),
        encoder_dims=256,
        decoder_dims=128,
        n_mels=80,
        fft_bins=80,
        postnet_dims=512,
        encoder_K=5,
        lstm_dims=1024,
        postnet_K=5,
        num_highways=4,
        dropout=0.5,
        stop_threshold=-3.4,
        speaker_embedding_size=256,
    ).to(device)

    model.load(syn_model_fpath)
    model.eval()

    r = int(model.r)
    metadata_fpath = in_dir / "train.txt"
    mel_dir = in_dir / "mels"
    embed_dir = in_dir / "embeds"

    dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir)
    data_loader = DataLoader(dataset, batch_size=4,
                             collate_fn=partial(collate_synthesizer, r=r),
                             num_workers=0)

    with open(out_dir / "synthesized.txt", "w") as file:
        for texts, mels, embeds, idx in tqdm(data_loader):
            texts, mels, embeds = texts.to(device), mels.to(device), embeds.to(device)
            _, mels_out, _, _ = model(texts, mels, embeds)

            for j, k in enumerate(idx):
                mel_out = mels_out[j].detach().cpu().numpy().T
                mel_out = mel_out[:int(dataset.metadata[k][4])]
                np.save(synth_dir / dataset.metadata[k][1], mel_out)
                file.write("|".join(dataset.metadata[k]) + "\n")


if __name__ == "__main__":
    generate_gta_mels()
