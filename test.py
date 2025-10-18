from pathlib import Path
from Encoder.audio import preprocess_wav
from Encoder.audio import wav_to_mel_spectrogram
import torch
from Tacotron.utils.text import *
from Encoder.model import SpeakerEncoder
from Tacotron.model import Tacotron
import numpy as np
from typing import Union, List
from wavernn.model import WaveRNN
import soundfile as sf
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint1=torch.load(r"Encoder/saved_models/libri_speaker_encoder/encoder.pt",map_location=device)
model1=SpeakerEncoder(device,device)
model1.load_state_dict(checkpoint1["model_state"])
checkpoint2=torch.load(r"Tacotron/utils/saved_model/tacotron_training/synthesizer.pt",map_location=device)
model2 = Tacotron(
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
model2.load_state_dict(checkpoint2["model_state"])
checkpoint3=torch.load(r"saved_models/vocoder_run_01/vocoder.pt",map_location=device)
model3 = WaveRNN(
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
model3.load_state_dict(checkpoint3["model_state"])
def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)

def synthesize_spectrograms(texts: List[str],
                                embeddings: Union[np.ndarray, List[np.ndarray]],
                                return_alignments=False):
        # Preprocess text inputs
        inputs = [text_to_sequence(text.strip(), ["english_cleaners"]) for text in texts]
        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        # Batch inputs
        batched_inputs = [inputs[i:i+16]
                             for i in range(0, len(inputs), 16)]
        batched_embeds = [embeddings[i:i+16]
                             for i in range(0, len(embeddings), 16)]

        specs = []
        for i, batch in enumerate(batched_inputs, 1):
            if True:
                print(f"\n| Generating {i}/{len(batched_inputs)}")

            # Pad texts so they are all the same length
            text_lens = [len(text) for text in batch]
            max_text_len = max(text_lens)
            chars = [pad1d(text, max_text_len) for text in batch]
            chars = np.stack(chars)

            # Stack speaker embeddings into 2D array for batch processing
            speaker_embeds = np.stack(batched_embeds[i-1])

            # Convert to tensor
            chars = torch.tensor(chars).long().to(device)
            speaker_embeddings = torch.tensor(speaker_embeds).float().to(device)

            # Inference
            _, mels, alignments = model2.generate(chars, speaker_embeddings)
            mels = mels.detach().cpu().numpy()
            for m in mels:
                # Trim silence from end of each spectrogram
                while np.max(m[:, -1]) < -3.4:
                    m = m[:, :-1]
                specs.append(m)

        if True:
            print("\n\nDone.\n")
        return (specs, alignments) if return_alignments else specs
def embed_frames_batch(frames_batch):
 
    frames = torch.from_numpy(frames_batch).to(device)
    embed = model1.forward(frames).detach().cpu().numpy()
    return embed


def compute_partial_slices(n_samples, partial_utterance_n_frames=160,
                           min_pad_coverage=0.75, overlap=0.5):
   
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1

    samples_per_frame = int((16000 * 10 / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices

def embed_utterance(wav, using_partials=True, return_partials=False, **kwargs):

    if not using_partials:
        frames = wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed

    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

    # Split the utterance into partials
    frames = wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(frames_batch)

    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed

input_file= Path(r"test.wav")
preprocessed_wav = preprocess_wav(input_file)
embed = embed_utterance(preprocessed_wav)


text="I am the captain of Indian Cricket Team.I love to play cricket."

mel=synthesize_spectrograms([text],[embed])[0]
mel = mel / 4.
mel = torch.from_numpy(mel[None, ...])
wav = model3.generate(mel, True, 8000, 800, True,  None)
generated_wav = np.pad(wav, (0, 16000), mode="constant")
generated_wav = preprocess_wav(generated_wav)
path= Path(r"output_final3.wav")
sf.write(path, wav.astype(np.float32), 16000)
