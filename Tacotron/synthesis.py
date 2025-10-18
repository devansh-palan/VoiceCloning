
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from model import Tacotron
from tacotron_dataset import SynthesizerDataset, collate_synthesizer
from utils.text import symbols


def run_synthesis(in_dir: Path, out_dir: Path, syn_model_fpath: Path):
    
    synth_dir = out_dir / "mels_gta"
    synth_dir.mkdir(exist_ok=True, parents=True)
   

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Synthesizer using device:", device)

   
    model = Tacotron(embed_dims=512,
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
                     dropout=0., # Use zero dropout for gta mels
                     stop_threshold=-3.4,
                     speaker_embedding_size=256).to(device)

    print("\nLoading weights at %s" % syn_model_fpath)
    model.load(syn_model_fpath)
    print("Tacotron weights loaded from step %d" % model.step)

    r = np.int32(model.r)

    model.eval()

    metadata_fpath = in_dir.joinpath("train.txt")
    mel_dir = in_dir.joinpath("mels")
    embed_dir = in_dir.joinpath("embeds")

    dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir)
    collate_fn = partial(collate_synthesizer, r=r)
    data_loader = DataLoader(dataset, 16, collate_fn=collate_fn, num_workers=2)
    
    meta_out_fpath = out_dir / "synthesized.txt"
    with meta_out_fpath.open("w") as file:
        for i, (texts, mels, embeds, idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
            texts, mels, embeds = texts.to(device), mels.to(device), embeds.to(device)

  
            if device.type == "cuda" and torch.cuda.device_count() > 1:
                _, mels_out, _ = data_parallel_workaround(model, texts, mels, embeds)
            else:
                _, mels_out, _, _ = model(texts, mels, embeds)

            for j, k in enumerate(idx):
                
                mel_filename = Path(synth_dir).joinpath(dataset.metadata[k][1])
                mel_out = mels_out[j].detach().cpu().numpy().T

                mel_out = mel_out[:int(dataset.metadata[k][4])]

           
                np.save(mel_filename, mel_out, allow_pickle=False)

                file.write("|".join(dataset.metadata[k]))
