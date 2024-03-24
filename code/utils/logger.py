import wandb
import numpy as np
from datetime import datetime


class Logger:
    def __init__(self, WANDB: bool, config: dict):
        self.config = config
        if WANDB:
            wandb.init(project='music', name='xylophone_experiment_' + datetime.now().strftime('%m-%d-%H-%M'), config=config)
            self.logger = self._logger_wandb
        else:
            self.logger = self._logger_terminal

    def log(self, info: dict):
        self.logger(info)

    def _logger_wandb(self, info: dict):
        wandb.log(info)
        return info
    
    def _logger_terminal(self, info: dict):
        print(info)
        return info
    
    def close(self):
        if self.config['logger'] == 'wandb':
            wandb.finish()

    def episode_reward_stat(self, reward_info: dict):
        mean_reward = np.mean(reward_info['episode_rewards'])
        std_reward = np.std(reward_info['episode_rewards'])
        min_reward = np.min(reward_info['episode_rewards'])
        max_reward = np.max(reward_info['episode_rewards'])

        mean_onset_reward = np.mean(reward_info['epi_onset_rewards'])
        mean_timing_reward = np.mean(reward_info['epi_timing_rewards'])
        mean_hitting_reward = np.mean(reward_info['epi_hit_rewards'])

        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'min_reward': min_reward,
            'max_reward': max_reward,
            'mean_onset_reward': mean_onset_reward,
            'mean_timing_reward': mean_timing_reward,
            'mean_hitting_reward': mean_hitting_reward
        }