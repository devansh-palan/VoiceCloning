import librosa
import librosa.filters
import numpy as np
from scipy import signal




def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def get_hop_size():
    hop_size = 200
    
    return hop_size


def melspectrogram(wav):
    D = _stft(preemphasis(wav, 0.97, True))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20
    
    if True:
      return _normalize(S)
    return S

def _stft(y):
    
    return librosa.stft(y=y, n_fft=800, hop_length=get_hop_size(), win_length=800)



_mel_basis = None
def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    # fmax<=sample_rate/2
    return librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=125, fmax=7600)


def _amp_to_db(x):
    min_level = np.exp(-100 / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    # clipping in normalization
    
    return np.clip((2 * 4) * ((S - (-100)) / (100)) - 4,
                           -4, 4)
    

