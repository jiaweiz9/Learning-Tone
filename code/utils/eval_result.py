import numpy as np
import librosa
import matplotlib.pyplot as plt
import wavio
from utils.reward_functions import assign_rewards_to_episode

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

def visualize_audio_wavio(ref_audio, audio_data, sr):
    time = np.arange(0, max(len(audio_data), len(ref_audio))) / sr
    ref_audio = np.pad(ref_audio, (0, len(time) - len(ref_audio)))
    audio_data = np.pad(audio_data, (0, len(time) - len(audio_data)))
    fig, axs = plt.subplots(2, figsize=(9, 12))

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

def visualize_reward_components(ref_audio, audio_data, epi_length, sr):
    amp_reward_list, dtw_reward_list, hit_timing_reward = assign_rewards_to_episode(ref_audio, audio_data, epi_length)
    time = np.arange(0, max(len(audio_data), len(ref_audio))) / sr
    ref_audio = np.pad(ref_audio, (0, len(time) - len(ref_audio)))
    audio_data = np.pad(audio_data, (0, len(time) - len(audio_data)))

    # Create a figure and two subplots with shared x-axis
    fig, axs = plt.subplots(4, figsize=(9, 12))

    # Plot the reference audio
    axs[0].plot(time, ref_audio, color='blue')
    axs[0].set_title('Reference Audio')
    axs[0].set_ylabel('Amplitude')

    # Plot the performance audio
    axs[1].plot(time, audio_data, color='orange')
    axs[1].set_title('Performance Audio')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')

    amp_coef = 1.0
    dtw_coef = 0.1
    hit_times_coef = 10.0
    hit_timing_coef = 10.0
    # Plot reward components
    axs[2].plot(range(epi_length), amp_reward_list * amp_coef, color='red', label='Amplitude Reward')
    axs[2].plot(range(epi_length), dtw_reward_list * dtw_coef, color='green', label='DTW Reward')
    # axs[2].plot(range(epi_length), hit_reward_list * hit_coef, color='purple', label='Hit Reward')
    axs[2].set_title('Reward Components')
    axs[2].set_ylabel('Reward')

    # Plot the total reward
    total_reward = amp_reward_list * amp_coef + dtw_reward_list * dtw_coef
    total_reward[-1] += hit_timing_coef * hit_timing_reward
    axs[3].plot(range(epi_length), total_reward, color='black', label='Total Reward')
    axs[3].set_title('Total Reward')

    plt.tight_layout()
    plt.show()

def save_vis_reward_components(ref_audio, audio_data, epi_length, sr, rewards_dict, img_path):
    component_type_count = len(rewards_dict)
    time = np.arange(0, max(len(audio_data), len(ref_audio))) / sr
    ref_audio = np.pad(ref_audio, (0, len(time) - len(ref_audio)))
    audio_data = np.pad(audio_data, (0, len(time) - len(audio_data)))

    # Create a figure and two subplots with shared x-axis
    fig, axs = plt.subplots(3, figsize=(9, 12))

    # Plot the reference audio
    axs[0].plot(time, ref_audio, color='blue')
    axs[0].set_title('Reference Audio')
    axs[0].set_ylabel('Amplitude')

    # Plot the performed audio
    axs[1].plot(time, audio_data, color='orange')
    axs[1].set_title('Performed Audio')
    axs[1].set_xlabel('Time (s)')

    available_colors = ['green', 'purple', 'red', 'black', 'pink', 'cyan', 'magenta', 'yellow', ]
    for i, (component_type, reward_list) in enumerate(rewards_dict.items()):
        if not isinstance(reward_list, np.ndarray):
            point_x = epi_length - 1
            point_y = reward_list
            axs[2].scatter(point_x, point_y, color=available_colors[i], label=component_type)
            continue

        elif len(reward_list) != epi_length:
            reward_list = np.pad(reward_list, (0, epi_length - len(reward_list)))
        axs[2].plot(range(epi_length), reward_list, color=available_colors[i], label=component_type, alpha=0.5)
        axs[2].set_title("Reward Components")
        axs[2].set_ylabel('Reward')
        axs[2].legend()

    plt.tight_layout()
    if img_path is None:
        plt.show()
    plt.savefig(img_path)


def clip_audio(audio_data, sr, start_time, end_time):
    start_frame = int(start_time * sr)
    end_frame = int(end_time * sr)
    return audio_data[start_frame:end_frame]


if __name__ == '__main__':
    # Load the reference audio and the performance audio
    ref_audio, sr = librosa.load('ref_audio/xylophone/ref_hit1.wav', sr=None)
    audio_data, sr = librosa.load('result/record_audios/ref_1.wav', sr=None)

    # Visualize the reference audio and the performance audio
    # visualize_audio(ref_audio, audio_data, sr)
    # visualize_reward_components(ref_audio, audio_data, 200, sr = sr)
    ref_audio = clip_audio(ref_audio, sr, 0, 4)
    visualize_audio_wavio(ref_audio, audio_data, sr)
    wavio.write("ref_audio/xylophone/ref_hit1_clip_4.wav", ref_audio, sr, sampwidth=4)