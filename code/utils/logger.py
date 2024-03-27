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
        stat_info = {}
        mean_reward = np.mean(reward_info['episode_rewards'])
        std_reward = np.std(reward_info['episode_rewards'])
        min_reward = np.min(reward_info['episode_rewards'])
        max_reward = np.max(reward_info['episode_rewards'])

        stat_info['mean_reward'] = mean_reward
        stat_info['std_reward'] = std_reward
        stat_info['min_reward'] = min_reward
        stat_info['max_reward'] = max_reward

        for key, value in reward_info['reward_components'].items():
            stat_info['mean_' + key] = np.mean(value)

        return stat_info