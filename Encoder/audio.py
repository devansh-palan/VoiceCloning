from pathlib import Path
from typing import Optional, Union
import numpy as np
import librosa as _librosa
import struct
from scipy.ndimage.morphology import binary_dilation
from warnings import warn
try:
    import webrtcvad
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad=None

int16_max = (2 ** 15) - 1


def wav_to_mel_spectrogram(wav):
    frames = _librosa.feature.melspectrogram(
        y=wav,
        sr=16000,
        n_fft=int(16000 * 25 / 1000),
        hop_length=int(16000 * 10 / 1000),
        n_mels=40
    )
    return frames.astype(np.float32).T


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None,
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = True):
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = _librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav if needed
    if source_sr is not None and source_sr != 16000:
        wav = _librosa.resample(y=wav, orig_sr=source_sr, target_sr=16000)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    if normalize:
        wav = normalize_volume(wav, -30, increase_only=True)
    if webrtcvad and trim_silence:
        wav = trim_long_silences(wav)
    
    return wav

def trim_long_silences(wav):
    # Compute the voice detection window size
    samples_per_window = (30 * 16000) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=16000))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, 8)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(6 + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))
