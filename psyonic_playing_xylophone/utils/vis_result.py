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
            self.logger.record_mean("onset_shape_reward", self.training_env.get_attr("last_onset_shape_reward")[0])
            self.logger.record_mean("amplitude_reward", self.training_env.get_attr("last_amplitude_reward")[0])

        if self.num_timesteps % 1000 == 0:
            last_rec_audio = self.training_env.get_attr("last_rec_audio")[0] # fix: returned value is a list
            ref_audio = self.training_env.get_attr("ref_audio")[0]
            self.__visualize_audio(ref_audio, last_rec_audio, sr=44100)

        return True

    def _on_rollout_start(self) -> None:
        # os.system("clear")
        print(f"Rollout {self.num_timesteps // 1000} started")
        return super()._on_rollout_start()


    def _on_rollout_end(self) -> None:
        return super()._on_rollout_end()


    def __visualize_audio(self, ref_audio, rec_audio, rec_idx, sr=44100) -> None:
        
        plt.figure()
        time = np.arange(0, len(rec_audio)) / 44100
        plt.plot(time, rec_audio, color='blue', alpha=0.3)

        ref_audio = np.pad(ref_audio, (0, len(rec_audio) - len(ref_audio)), 'constant')
        plt.plot(time, ref_audio, color='red', alpha=0.3)

        rec_idx = rec_idx * 0.02
        plt.scatter(rec_idx, np.zeros_like(rec_idx), color='black', marker='x')

        # plt.title(f'Episode {rec_idx} - Reference Audio (red) vs. Recorded Audio (blue)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend(['Recorded Audio', 'Reference Audio'])

        file_name = f"episode_{self.num_timesteps}.png"
        img_path = os.path.join(self.figures_path, file_name)
        plt.savefig(img_path)
        plt.close()