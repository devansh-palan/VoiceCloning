from pathlib import Path
from functools import partial
from datetime import datetime
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from model import Tacotron
from tacotron_dataset import SynthesizerDataset, collate_synthesizer
from utils.text import symbols

def np_now(x): return x.detach().cpu().numpy()
def time_string(): return datetime.now().strftime("%Y-%m-%d %H:%M")

def train(run_id, syn_dir, models_dir, force_restart):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_dir = models_dir / run_id
    model_dir.mkdir(exist_ok=True, parents=True)
    weights_fpath = model_dir / "synthesizer.pt"

    # Init model
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
        speaker_embedding_size=256
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if not force_restart and weights_fpath.exists():
        print(f"Loading model from {weights_fpath}")
        model.load(weights_fpath, optimizer)
    else:
        print("Starting from scratch.")
        model.save(weights_fpath)

    # Dataset
    dataset = SynthesizerDataset(
        metadata_fpath=syn_dir / "train.txt",
        mel_dir=syn_dir / "mels",
        embed_dir=syn_dir / "embeds"
    )

    data_loader = DataLoader(
        dataset,
        batch_size=12,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(collate_synthesizer, r=1)
    )

    model.r = 1
    max_steps = 2000

    # Training loop
    while model.get_step() < max_steps:
        for i, (texts, mels, embeds, idx) in enumerate(data_loader):
            step = model.get_step()
            if step >= max_steps:
                break

            # Generate stop token targets
            stop = torch.ones(mels.shape[0], mels.shape[2])
            for j, k in enumerate(idx):
                stop[j, :int(dataset.metadata[k][4]) - 1] = 0

            texts, mels, embeds, stop = texts.to(device), mels.to(device), embeds.to(device), stop.to(device)

            # Forward pass
            m1_hat, m2_hat, _, stop_pred = model(texts, mels, embeds)

            # Loss
            m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
            m2_loss = F.mse_loss(m2_hat, mels)
            stop_loss = F.binary_cross_entropy(stop_pred, stop)
            loss = m1_loss + m2_loss + stop_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            if step % 10 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")

            if step % 100 == 0:
                model.save(weights_fpath, optimizer)

    # Save at the end of training
    print("Training complete. Saving final model...")
    model.save(weights_fpath, optimizer)
