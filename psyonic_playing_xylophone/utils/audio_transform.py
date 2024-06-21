import numpy as np
from numpy.typing import ArrayLike
import librosa
import scipy
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import os
import time

def waveform_to_frequence(audio: ArrayLike) -> ArrayLike:
    return fftshift(fft(audio)), fftshift(fftfreq(len(audio), 44100))


def waveform_to_time_freq(audio: ArrayLike, to_db=False) -> ArrayLike:
    S = librosa.stft(audio / np.max(audio))
    if to_db:
        return librosa.amplitude_to_db(np.abs(S))
    return np.abs(S)


def waveform_to_Hilbert_trans(audio: ArrayLike) -> ArrayLike:
    return scipy.signal.hilbert(audio)


def display_spectrogram(rec_audio: ArrayLike, ref_audio: ArrayLike, sr: int):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    img = librosa.display.specshow(waveform_to_time_freq(rec_audio), y_axis='log', x_axis='time', sr=sr, ax=ax[0])

    librosa.display.specshow(waveform_to_time_freq(ref_audio), y_axis='log', x_axis='time', sr=sr, ax=ax[1])

    ax[0].set(title='Linear-frequency power spectrogram')
    # ax[0].label_outer()

    ax[1].set(title='Log-frequency power spectrogram')
    # ax[1].label_outer()
    # fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()


def display_freq_components(rec_audio: ArrayLike, ref_audio: ArrayLike, sr: int = 44100):
    rec_freq_comp, freqs = waveform_to_frequence(rec_audio)
    print(len(rec_freq_comp), len(freqs))
    ref_freq_comp, freqs = waveform_to_frequence(ref_audio)
    print(len(ref_freq_comp), len(freqs))

    freq_range = np.arange(0, len(rec_freq_comp))
    # import pdb 
    # print(np.real(rec_freq_comp))
    # pdb.set_trace()
    # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    rec_freq_comp = rec_freq_comp[len(rec_freq_comp) // 2:] * 2 / len(rec_freq_comp)
    ref_freq_comp = ref_freq_comp[len(ref_freq_comp) // 2:] * 2 / len(ref_freq_comp)
    freqs = freqs[len(freqs) // 2:]

    plt.plot(freqs, np.abs(rec_freq_comp), color='blue', alpha=0.3)
    plt.plot(freqs, np.abs(ref_freq_comp), color='red', alpha=0.3)

    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.legend(['Recorded Audio', 'Reference Audio'])
    # plt.plot(freqs, (np.abs(rec_freq_comp) - np.abs(ref_freq_comp)), color='green', alpha=0.3)
    plt.show()
    print(np.sum(np.abs(np.abs(rec_freq_comp) - np.abs(ref_freq_comp))))


def display_audio(ref_audio, rec_audio, sr=44100) -> None:
        
    plt.figure(figsize=(20, 6))

    max_len = max(len(rec_audio), len(ref_audio))
    ref_audio = np.pad(ref_audio, (0, max_len - len(ref_audio)), 'constant')
    rec_audio = np.pad(rec_audio, (0, max_len - len(rec_audio)), 'constant')

    time = np.arange(0, len(rec_audio)) / sr
    plt.plot(time, rec_audio, color='blue', alpha=0.3)
    plt.plot(time, ref_audio, color='red', alpha=0.3)

    # rec_idx = rec_idx * 0.02
    # plt.scatter(rec_idx, np.zeros_like(rec_idx), color='black', marker='x')

    # plt.title(f'Episode {rec_idx} - Reference Audio (red) vs. Recorded Audio (blue)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(['Recorded Audio', 'Reference Audio'])

    # file_name = f"episode_{self.num_timesteps}.png"
    # img_path = os.path.join(self.figures_path, file_name)
    # plt.savefig(img_path)
    # plt.close()
    plt.show()


def dtw_similarity(rec_audio: ArrayLike, ref_audio: ArrayLike) -> float:
    diff, _ = fastdtw(rec_audio / (np.max(rec_audio)), ref_audio / np.max(ref_audio), radius=5)
    return np.sum(diff)

def pre_emphasis_high_pass(audio, alpha=0.95):
    filtered_data = np.copy(audio)

    for n in range(1, len(audio)):
        # filtered_data[n] = alpha * audio[n] + (1 - alpha) * audio[n - 1]  # low-pass
        filtered_data[n] = audio[n] - alpha * audio[n - 1] # high-pass
    
    return filtered_data



if __name__ == "__main__":
    rec_audio_path = "results/audios/0620_1813-sn7xeq3p/episode_22000.wav"
    ref_audio_path = "ref_audio/xylophone_keyB/amp045_clip.wav"
    rec_audio, sr = librosa.load(path=rec_audio_path)
    ref_audio, sr = librosa.load(path=ref_audio_path)

    filtered_rec_audio = pre_emphasis_high_pass(rec_audio)
    filtered_ref_audio = pre_emphasis_high_pass(ref_audio)
    # print(dtw_similarity(rec_audio, ref_audio))
    # display_spectrogram(rec_audio, ref_audio, sr)
    # display_freq_components(rec_audio / np.max(rec_audio), ref_audio / np.max(ref_audio))
    display_freq_components(filtered_rec_audio[:88200] / np.max(filtered_rec_audio), filtered_ref_audio[:88200] / np.max(filtered_ref_audio))
    # amp_freqs = np.abs(waveform_to_frequence(ref_audio))
    # print(len(amp_freqs))
    # for i in range(len(amp_freqs)):
    #     print(amp_freqs[i])
    # print(amp_freqs)