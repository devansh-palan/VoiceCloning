
from pathlib import Path

import numpy as np
from tqdm import tqdm

import audio  


_AUDIO_EXTENSIONS = ("wav", "flac", "m4a", "mp3")


def _preprocess_speaker(speaker_dir: Path, dataset_root: Path, out_dir: Path, skip_existing: bool):
    
    speaker_name = "_".join(speaker_dir.relative_to(dataset_root).parts)
    speaker_out_dir = out_dir.joinpath(speaker_name)
    speaker_out_dir.mkdir(parents=True, exist_ok=True)
    sources_fpath = speaker_out_dir.joinpath("_sources.txt")

    if sources_fpath.exists() and skip_existing:
        with sources_fpath.open("r") as f:
            existing_fnames = {line.split(",")[0] for line in f}
    else:
        existing_fnames = set()

    sources_file = sources_fpath.open("a" if skip_existing else "w")

    for ext in _AUDIO_EXTENSIONS:
        for in_fpath in speaker_dir.glob(f"**/*.{ext}"):
            out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts).replace(f".{ext}", ".npy")
            if skip_existing and out_fname in existing_fnames:
                continue

            try:
                wav = audio.preprocess_wav(in_fpath)
                if len(wav) == 0:
                    continue

                frames = audio.wav_to_mel_spectrogram(wav)
                if len(frames) < 160:
                    continue

                out_fpath = speaker_out_dir.joinpath(out_fname)
                np.save(out_fpath, frames)
                sources_file.write(f"{out_fname},{in_fpath}\n")
            except Exception as e:
                print(f"Failed to process {in_fpath}: {e}")

    sources_file.close()



def _preprocess_speaker_dirs(speaker_dirs,dataset_root,out_dir,skip_existing):
    print(f"Preprocessing {len(speaker_dirs)} speaker subdirs.")

    
    for speaker_dir in tqdm(speaker_dirs, total=len(speaker_dirs), unit="dirs"):
        _preprocess_speaker(
            speaker_dir=speaker_dir,
            dataset_root=dataset_root,
            out_dir=out_dir,
            skip_existing=skip_existing
        )

    print("Done preprocessing.\n")



def preprocess_librispeech(dataset_root: Path, out_dir: Path, skip_existing=False):
    if not dataset_root.exists():
        print(f"Dataset path {dataset_root} does not exist.")
        return

    
    speaker_dirs = [d for d in dataset_root.glob("*/*") if d.is_dir()]
    _preprocess_speaker_dirs(speaker_dirs, dataset_root, out_dir, skip_existing)
