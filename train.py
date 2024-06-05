import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf
from psyonic_playing_xylophone.envs import PsyonicThumbEnv
from psyonic_playing_xylophone.utils.vis_result import VisualizeEpisodeCallback
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from datetime import datetime
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id: str, config: dict, seed: int = 0):
    """
    Utility function for DummyVec env. First make the environment and then flatten it
    """
    def _init():
        env = gym.make(env_id, config = config)
        env = gym.wrappers.FlattenObservation(env)
        env.reset(seed=seed)
        return env
    set_random_seed(seed)
    return _init


class TrainPPO:
    def __init__(self, config):
        self.env_id = config["env_id"]
        self.policy = config["policy"]
        self.total_timesteps = config["total_timesteps"]
        # self.reset_num_timesteps = config["reset_num_timesteps"]
        self.load_model_path = config.get("load_model_path", None)
        self.n_steps_per_update = config["n_steps_per_update"]
        self.epi_length = config["epi_length"]

        # self.env = gym.make(self.env_id, config = config)
        # self.env = gym.wrappers.FlattenObservation(self.env)
        # self.dummy_vec_env = DummyVecEnv([make_env(self.env_id, config)])
        self.dummy_vec_env = make_vec_env(
            self.env_id, 
            n_envs=1, 
            seed=0, 
            wrapper_class=gym.wrappers.FlattenObservation, 
            env_kwargs={"config": config},
            # monitor_kwargs={
            #     "stats_window_size": 100
            #     }
            )
        
        self.dummy_vec_env = VecNormalize(
            self.dummy_vec_env, 
            norm_obs=True, 
            norm_reward=True, 
            clip_obs=10.
            )

        # Prepare the environment for training
        # check_env(self.env)

        # Initialize Wandb
        self.wandb_run = wandb.init(
            project="Psyonic_Playing_Xylophone-sb3",
            config=config,
            sync_tensorboard=True,
            group="thumb_sb3",
            tags=["ppo"],
            resume="allow",
            id=config.get("wandb_run_id", None) # Note: We can continue the logging in wandb, but cannot do the same in other results logging (tensorboard, figures, models, etc.)
        )
        self.results_folder_name = f"{datetime.now().strftime('%m%d_%H%M')}-{self.wandb_run.id}"

        # Set up WandbCallback to load training progress to wandb
        self.wandb_callback = WandbCallback(
            # gradient_save_freq=1000,
            # model_save_freq=10000,
            # model_save_path=f"./results/ppo/{self.results_folder_name}",
            verbose=1,
            # log="parameters",
        )

        # Set up checkpoint callback to save model
        self.checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"./results/ppo/{self.results_folder_name}",
            name_prefix='model'
        )

        self.visualize_callback = VisualizeEpisodeCallback(
            figures_path=f"./results/figures/{ self.results_folder_name}"
        )
        self.__load_model()

      
    def train(self):
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=[
                self.checkpoint_callback, 
                self.wandb_callback, 
                self.visualize_callback
            ],
        )

        wandb.finish()
    
    # def save_model(self, path: str = None):
    #     if path is None:
    #         path = self.config["model_path"]
    #     self.model.save(path)

    def __load_model(self):
        if self.load_model_path is None:
            self.model = PPO(
                policy=self.policy, 
                n_steps=self.n_steps_per_update, # number of steps to run for each environment per update, 
                env=self.dummy_vec_env,
                tensorboard_log=f"./results/tensorboard/{self.results_folder_name}",
                verbose=1,
                # stats_window_size=self.n_steps / self.epi_length, # compute rollout statistics over the last iteration
            )
        else:
            import os
            folder_path, model_file = os.path.split(self.load_model_path)
            model_name = os.path.splitext(model_file)[0]
            prefix, num_timesteps, _ = model_name.split("_")

            self.model = PPO.load(self.load_model_path, env=self.dummy_vec_env)
    


@hydra.main(version_base=None, config_path="psyonic_playing_xylophone/conf/psyonic_thumb", config_name="train")
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