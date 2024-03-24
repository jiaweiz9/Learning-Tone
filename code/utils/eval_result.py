import numpy as np
import librosa
import matplotlib.pyplot as plt
import wavio

#TODO: Visualize the reference audio and the performed audio
def visualize_audio(ref_audio, audio_data, sr):
    time = np.arange(0, max(len(audio_data), len(ref_audio))) / sr
    ref_audio = np.pad(ref_audio, (0, len(time) - len(ref_audio)))
    audio_data = np.pad(audio_data, (0, len(time) - len(audio_data)))

    # Create a figure and two subplots with shared x-axis
    fig, axs = plt.subplots(2, sharex=True, figsize=(12, 8))

    # Plot the reference audio
    axs[0].plot(time, ref_audio, color='blue')
    axs[0].set_title('Reference Audio')
    axs[0].set_ylabel('Amplitude')

    # Plot the performance audio
    axs[1].plot(time, audio_data, color='orange')
    axs[1].set_title('Performance Audio')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Load the reference audio and the performance audio
    ref_audio, sr = librosa.load('ref_audio/xylophone/ref_hit2.wav', sr=None)
    audio_data, sr = librosa.load('result/record_audios/episode_199.wav', sr=None)

    # Visualize the reference audio and the performance audio
    visualize_audio(ref_audio, audio_data, sr)