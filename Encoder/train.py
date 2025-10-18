from pathlib import Path
import torch

from speaker_verification_dataset import SpeakerVerificationDataset, SpeakerVerificationDataLoader
from model import SpeakerEncoder


def main():
   
    run_id = "libri_speaker_encoder"
    clean_data_root = Path(r"../LibriSpeech/processed/encoder/")
    models_dir = Path("saved_models")
    force_restart = True
    max_steps = 5000
    save_interval = 100 
    
    
    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset, 
        32, 
        10, 
        num_workers=0  
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = device

    
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

   
    model_dir = models_dir / run_id
    model_dir.mkdir(exist_ok=True, parents=True)
    state_fpath = model_dir / "encoder.pt"
    start_step = 0
    if(state_fpath.exists()):
        checkpoint=torch.load(state_fpath,map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step= checkpoint["step"]
        print(f"Resuming training from step {start_step}.")
    else:    
     print("Starting training from scratch.")
    model.train()

    # Training loop
    for step, speaker_batch in enumerate(loader,start=start_step):
        if step >= max_steps:
            break

        inputs = torch.from_numpy(speaker_batch.data).to(device)
        embeds = model(inputs)
        embeds_loss = embeds.view((32, 10, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)

        model.zero_grad()
        loss.backward()
        model.do_gradient_ops()
        optimizer.step()

        print(f"Step {step} | Loss: {loss.item():.4f} | EER: {eer:.4f}")

        if (step + 1) % save_interval == 0:
            checkpoint_path = model_dir / f"encoder_step_{step+1}.pt"
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at step {step+1} to {checkpoint_path}")

    
    torch.save({
        "step": step + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, state_fpath)
    print(f"\nTraining complete. Final model saved to: {state_fpath}")
if __name__ == "__main__":
    
    main()