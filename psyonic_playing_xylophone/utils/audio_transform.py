import numpy as np
from numpy.typing import ArrayLike
import librosa
import scipy
import matplotlib.pyplot as plt

def waveform_to_frequence(audio: ArrayLike) -> ArrayLike:
    return scipy.fft.fft(audio)

def waveform_to_time_freq(audio: ArrayLike, to_db=False) -> ArrayLike:
    S = librosa.stft(audio / np.max(audio))
    if to_db:
        return librosa.amplitude_to_db(np.abs(S))
    return S

def waveform_to_Hilbert_trans(audio: ArrayLike) -> ArrayLike:
    return scipy.signal.hilbert(audio)


if __name__ == "__main__":
    rec_audio_path = ""
    ref_audio_path = ""
    rec_audio, sr = librosa.load(path=rec_audio_path)
    ref_audio, sr = librosa.load(path=ref_audio_path)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    librosa.display.specshow(waveform_to_time_freq(rec_audio), y_axis='log', x_axis='time', sr=sr, ax=ax[0])

    librosa.display.specshow(waveform_to_time_freq(ref_audio), y_axis='log', x_axis='time', sr=sr, ax=ax[1])