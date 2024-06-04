import numpy as np
import librosa
import matplotlib.pyplot as plt
import wavio
from stable_baselines3.common.callbacks import BaseCallback
import os
# from utils.reward_functions import assign_rewards_to_episode

class VisualizeEpisodeCallback(BaseCallback):
    def __init__(self, verbose: int = 0, figures_path: str = None):
        super().__init__(verbose)
        # self.unwrapped_env = self.training_env.unwrapped
        if figures_path is None:
            self.figures_path = "results/figures/"
        else:
            self.figures_path = figures_path

        if not os.path.exists(self.figures_path):
            os.makedirs(self.figures_path)
        

    def _on_step(self) -> bool:
        """
        Visualize the latest episode data every 1000 timesteps, i.e., 10 episodes
        """
        # print(self.training_env)
        if self.num_timesteps % 100 == 0:
            self.logger.record_mean("hitting_times_reward", self.training_env.get_attr("last_hitting_times_reward")[0])
            self.logger.record_mean("hitting_timing_reward", self.training_env.get_attr("last_hitting_timing_reward")[0])

        if self.num_timesteps % 500 == 0:
            last_rec_audio = self.training_env.get_attr("last_rec_audio")[0] # fix: returned value is a list
            ref_audio = self.training_env.get_attr("ref_audio")[0]
            self.__visualize_audio(ref_audio, last_rec_audio, sr=44100)

        return True

    def _on_rollout_start(self) -> None:
        os.system("clear")
        print(f"Rollout {self.num_timesteps // 100} started")
        return super()._on_rollout_start()


    def _on_rollout_end(self) -> None:
        return super()._on_rollout_end()


    def __visualize_audio(self, ref_audio, rec_audio, sr=44100) -> None:
        # component_type_count = len(rewards_dict)/
        time = np.arange(0, max(len(rec_audio), len(ref_audio))) / sr
        ref_audio = np.pad(ref_audio, (0, len(time) - len(ref_audio)))
        rec_audio = np.pad(rec_audio, (0, len(time) - len(rec_audio)))

        print("time shape: ", time.shape)
        print("ref_audio shape: ", ref_audio.shape)
        print("rec_audio shape: ", rec_audio.shape)

        # Create a figure and two subplots with shared x-axis
        fig, axs = plt.subplots(2, figsize=(9, 12))

        # Plot the reference audio
        axs[0].plot(time, ref_audio, color='blue')
        axs[0].set_title('Reference Audio')
        axs[0].set_ylabel('Amplitude')

        # Plot the performed audio
        axs[1].plot(time, rec_audio, color='orange')
        axs[1].set_title('Performed Audio')
        axs[1].set_xlabel('Time (s)')

        available_colors = ['red', 'purple', 'black', 'pink', 'cyan', 'magenta', 'yellow', ]
        # for i, (component_type, reward_list) in enumerate(rewards_dict.items()):
        #     if not isinstance(reward_list, np.ndarray):
        #         point_x = epi_length - 1
        #         point_y = reward_list
        #         axs[2].scatter(point_x, point_y, color=available_colors[i], label=component_type)
        #         continue

        #     elif len(reward_list) != epi_length:
        #         reward_list = np.pad(reward_list, (0, epi_length - len(reward_list)))
        #     axs[2].plot(range(epi_length), reward_list, color=available_colors[i], label=component_type, alpha=0.5)
        #     axs[2].set_title("Reward Components")
        #     axs[2].set_ylabel('Reward')
        #     axs[2].legend()
        file_name = f"episode_{self.num_timesteps}.png"
        img_path = os.path.join(self.figures_path, file_name)
        plt.savefig(img_path)