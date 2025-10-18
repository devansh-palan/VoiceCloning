import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from .display import stream
from .gen_wavernn import gen_testset
from .model import WaveRNN
from .datasets import VocoderDataset, collate_vocoder


def train(run_id: str, syn_dir: Path, voc_dir: Path, models_dir: Path, force_restart: bool):
    
    assert np.cumprod((5, 5, 8))[-1] == 200

    print("Initializing the model...")
    model = WaveRNN(
        rnn_dims=512,
        fc_dims=512,
        bits=9,
        pad=2,
        upsample_factors=(5, 5, 8),
        feat_dims=80,
        compute_dims=128,
        res_out_dims=128,
        res_blocks=10,
        hop_length=200,
        sample_rate=16000,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters())
    for p in optimizer.param_groups:
        p["lr"] = 1e-4
    loss_func = F.cross_entropy 

    model_dir = models_dir / run_id
    model_dir.mkdir(exist_ok=True)
    weights_fpath = model_dir / "vocoder.pt"
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of WaveRNN from scratch\n")
        model.save(weights_fpath, optimizer)
    else:
        print("\nLoading weights at %s" % weights_fpath)
        model.load(weights_fpath, optimizer)
        print("WaveRNN weights loaded from step %d" % model.step)

    # Initialize the dataset
    metadata_fpath = voc_dir.joinpath("synthesized.txt")
    mel_dir = voc_dir.joinpath("mels_gta")
    wav_dir = syn_dir.joinpath("audio")
    dataset = VocoderDataset(metadata_fpath, mel_dir, wav_dir)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for epoch in range(1, 75):
        data_loader = DataLoader(dataset, 100, shuffle=True, num_workers=0, collate_fn=collate_vocoder)
        start = time.time()
        running_loss = 0.

        for i, (x, y, m) in enumerate(data_loader, 1):
            if torch.cuda.is_available():
                x, m, y = x.cuda(), m.cuda(), y.cuda()

            # Forward pass
            y_hat = model(x, m)
            
            y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            
            y = y.unsqueeze(-1)

            # Backward pass
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            speed = i / (time.time() - start)
            avg_loss = running_loss / i

            step = model.get_step()
            k = step // 1000


            if step % 500 == 0 :
                 torch.save({
                     "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(),
            }, f"{model_dir}/step_{i}.pt")

            msg = f"| Epoch: {epoch} ({i}/{len(data_loader)}) | " \
                f"Loss: {avg_loss:.4f} | {speed:.1f} " \
                f"steps/s | Step: {k}k | "
            stream(msg)
           
        model.save(weights_fpath,optimizer)
        
        print(f"Model is saved after epochs : {epoch}") 

        gen_testset(model, test_loader, 2, True,
                    8000, 400, model_dir)
        print("")
