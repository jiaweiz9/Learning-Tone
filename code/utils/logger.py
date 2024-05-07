import wandb
import numpy as np
from datetime import datetime
import prettytable as pt


class Logger:
    def __init__(self, WANDB: bool, config: dict, resume: bool):
        self.config = config
        if WANDB:
            wandb.init(project='music_finger', name='xylophone_experiment_' + datetime.now().strftime('%m-%d-%H-%M'), config=config, resume=resume)
        #     self.logger = self._logger_wandb
        # else:
        #     self.logger = self._logger_terminal

    def log(self, info: dict, log_wandb: bool = False):
        if log_wandb:
            self._logger_wandb(info)
        else:
            self._logger_terminal(info)


    def _logger_wandb(self, info: dict):
        wandb.log(info)
        return info
    
    def _logger_terminal(self, info: dict):
        # print(info)
        table = pt.PrettyTable()
        for key, value in info.items():
            table.add_row([key, value])
        print(table)
        return info
    
    def close(self):
        if self.config['logger'] == 'wandb':
            wandb.finish()

    def episode_reward_stat(self, reward_info: dict):
        stat_info = {}
        mean_reward = np.mean(reward_info['episode_rewards'])

        stat_info['mean_reward'] = mean_reward

        for key, value in reward_info['reward_components'].items():
            stat_info['mean_' + key] = np.mean(value)

        return stat_info