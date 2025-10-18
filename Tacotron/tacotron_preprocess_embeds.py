from preprocess import create_embeddings

from pathlib import Path

synthesizer_root = Path(r"LibriSpeech/processed/synthesizer")
encoder_model_fpath = Path(r"Encoder/saved_models/libri_speaker_encoder/encoder.pt")
create_embeddings(
    synthesizer_root=synthesizer_root,
    encoder_model_fpath=encoder_model_fpath,
)
