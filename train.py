from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf
from psyonic_playing_xylophone.envs import PsyonicThumbEnv
from psyonic_playing_xylophone.utils.vis_result import VisualizeEpisodeCallback
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from datetime import datetime


class TrainPPO:
    def __init__(self, config):
        self.env_id = config["env_id"]
        self.policy = config["policy"]
        self.total_timesteps = config["total_timesteps"]
        # self.reset_num_timesteps = config["reset_num_timesteps"]
        self.load_model_path = config.get("load_model_path", None)
        self.n_steps = config["n_steps"]

        self.env = gym.make(self.env_id, config = config)
        self.env = gym.wrappers.FlattenObservation(self.env)

        # Prepare the environment for training
        # check_env(self.env)

        # Set up checkpoint callback to save model
        self.checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"./results/ppo/{datetime.now().strftime('%m-%d-%H-%M')}",
            name_prefix='rl_model'
        )

        # Initialize Wandb
        self.wandb_run = wandb.init(
            project="music_finger",
            config=config,
            sync_tensorboard=False,
            group="thumb_sb3",
            tags=["ppo"],
            resume="allow",
            id=config.get("wandb_run_id", None)
        )
        # Set up WandbCallback to load training progress to wandb
        self.wandb_callback = WandbCallback(
            gradient_save_freq=1000,
            model_save_freq=10000,
            model_save_path=f"./results/ppo/{datetime.now().strftime('%m-%d-%H-%M')}",
            verbose=1,
            # log="parameters",
        )

        self.visualize_callback = VisualizeEpisodeCallback()
        self.__load_model()

      
    def train(self):
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=[self.checkpoint_callback, self.wandb_callback, self.visualize_callback],
        )
    
    # def save_model(self, path: str = None):
    #     if path is None:
    #         path = self.config["model_path"]
    #     self.model.save(path)

    def __load_model(self):
        if self.load_model_path is None:
            self.model = PPO(
                policy=self.policy, 
                n_steps=self.n_steps, # number of steps to run for each environment per update
                env=self.env,
                verbose=1
            )
        else:
            import os
            folder_path, model_file = os.path.split(self.load_model_path)
            model_name = os.path.splitext(model_file)[0]
            prefix, num_timesteps, _ = model_name.split("_")

            self.model = PPO(
                policy=self.policy, 
                n_steps=self.n_steps, # number of steps to run for each environment per update
                env=self.env,
                verbose=1
            ).load(self.load_model_path)
    


@hydra.main(version_base=None, config_path="psyonic_playing_xylophone/conf", config_name="psyonic_thumb")
def launch_train(cfg: DictConfig):
    # print(cfg)
    cfg = OmegaConf.to_container(cfg)
    trainer = TrainPPO(config=cfg)

    trainer.train()


if __name__ == "__main__":
    test_config = {
    "psyonic": {
        "initial_state": [0, 0, 0, 0, 0],
        "min_degree": -90,
        "max_degree": 90,
        },
    "reward_weight": {
        "amplitude": 0.25,
        "hitting_times": 0.25,
        "onset_shape": 0.25,
        "hitting_timing": 0.25
        },
    "epi_length": 100,
    "reference_audio": "ref_audio/ref_high.wav"
    }
    
    launch_train()