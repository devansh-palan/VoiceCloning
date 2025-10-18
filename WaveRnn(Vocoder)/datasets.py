from torch.utils.data import Dataset
from pathlib import Path
from . import audio
import numpy as np
import torch


class VocoderDataset(Dataset):
    def __init__(self, metadata_fpath: Path, mel_dir: Path, wav_dir: Path):
        print("Using inputs from:\n\t%s\n\t%s\n\t%s" % (metadata_fpath, mel_dir, wav_dir))
        
        with metadata_fpath.open("r") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]
        
        gta_fnames = [x[1] for x in metadata if int(x[4])]
        gta_fpaths = [mel_dir.joinpath(fname) for fname in gta_fnames]
        wav_fnames = [x[0] for x in metadata if int(x[4])]
        wav_fpaths = [wav_dir.joinpath(fname) for fname in wav_fnames]
        self.samples_fpaths = list(zip(gta_fpaths, wav_fpaths))
        
        print("Found %d samples" % len(self.samples_fpaths))
    
    def __getitem__(self, index):  
        mel_path, wav_path = self.samples_fpaths[index]
        
        mel = np.load(mel_path).T.astype(np.float32) / 4.
        

        wav = np.load(wav_path)
        
        wav = audio.pre_emphasis(wav)
        wav = np.clip(wav, -1,1)
        
        r_pad =  (len(wav) // 200 + 1) * 200 - len(wav)
        wav = np.pad(wav, (0, r_pad), mode='constant')
        assert len(wav) >= mel.shape[1] * 200
        wav = wav[:mel.shape[1] * 200]
        assert len(wav) % 200 == 0
     
        
        
        quant = audio.encode_mu_law(wav, mu=2 ** 9)
        
        return mel.astype(np.float32), quant.astype(np.int64)

    def __len__(self):
        return len(self.samples_fpaths)
        
        
def collate_vocoder(batch): 
    mel_win = (200*5) // 200 + 2 * 2
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * 2) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + 2) * 200 for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + (200*5) + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()

    x = labels[:, :1000]
    y = labels[:, 1:]

    bits = 9

    x = audio.label_2_float(x.float(), bits)

    return x, y, mels