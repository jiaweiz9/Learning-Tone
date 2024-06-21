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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
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
        self.n_epochs = config["n_epochs"]
        # self.reset_num_timesteps = config["reset_num_timesteps"]
        self.load_model_path = config.get("load_model_path", None)
        self.n_steps_per_update = config["n_steps_per_update"]
        self.epi_length = config["epi_length"]
        self.no_wandb = config.get("no_wandb", False)
        self.wandb_run_id = config.get("wandb_run_id", None)
        self.device = config.get("device", "auto")
        self.learning_rate = config["learning_rate"]
        self.ent_coef = config["ent_coef"]

        # self.env = gym.make(self.env_id, config = config)
        # self.env = gym.wrappers.FlattenObservation(self.env)
        # self.dummy_vec_env = DummyVecEnv([make_env(self.env_id, config)])
        self.train_vec_env = self._make_normed_vec_env(config)
        self.eval_vec_env = self._make_normed_vec_env(config)

        # Prepare the environment for training
        # check_env(self.env)

        # Initialize Wandb
        if not self.no_wandb:
            self.wandb_run = wandb.init(
                project="Psyonic_Playing_Xylophone-sb3",
                config=config,
                sync_tensorboard=True,
                group="thumb_wrist_sb3",
                tags=["ppo"],
                resume="allow",
                id=self.wandb_run_id # Note: We can continue the logging in wandb, but cannot do the same in other results logging (tensorboard, figures, models, etc.)
            )

            # Set up WandbCallback to load training progress to wandb
            self.wandb_callback = WandbCallback(
                # gradient_save_freq=1000,
                # model_save_freq=10000,
                # model_save_path=f"./results/ppo/{self.results_folder_name}",
                verbose=1,
                # log="parameters",
            )

        self.results_folder_name = f"{datetime.now().strftime('%m%d_%H%M')}-{self.wandb_run.id}" if not self.no_wandb else f"{datetime.now().strftime('%m%d_%H%M')}-{self.wandb_run_id}"

        # Set up checkpoint callback to save model
        self.checkpoint_callback = CheckpointCallback(
            save_freq=self.n_steps_per_update,
            save_path=f"./results/ppo/{self.results_folder_name}",
            name_prefix='model',
            save_vecnormalize=True,
        )

        self.visualize_callback = VisualizeEpisodeCallback(
            figures_path=f"./results/figures/{ self.results_folder_name}",
            visualize_freq=self.n_steps_per_update,
            folder_name = self.results_folder_name
        )

        self.eval_callback = EvalCallback(
            eval_env=self.eval_vec_env,
            n_eval_episodes=1,
            best_model_save_path=f"./results/eval/{self.results_folder_name}",
            eval_freq=config["eval_freq"],
            deterministic=True,
            render=False,
        )
        
        self.__load_model()

    def _make_normed_vec_env(self, config):
        dummy_vec_env = make_vec_env(
            self.env_id, 
            n_envs=1, 
            seed=0, 
            wrapper_class=gym.wrappers.FlattenObservation, 
            env_kwargs={
                "config": config,
                "render_mode": None,
                },
            # monitor_kwargs={
            #     "stats_window_size": 100
            #     }
            )
        
        # normed_vec_env = VecNormalize(
        #     dummy_vec_env, 
        #     norm_obs=True, 
        #     norm_reward=True, 
        #     clip_obs=10.
        #     )
        return dummy_vec_env
      
    def train(self):
        if not self.no_wandb:
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=[
                    self.checkpoint_callback, 
                    self.wandb_callback, 
                    self.visualize_callback,
                    self.eval_callback,
                ],
            )
        else:
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=[
                    self.checkpoint_callback,
                    self.visualize_callback,
                    self.eval_callback,
                ],
            )

        if not self.no_wandb:
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
                learning_rate=self.learning_rate,
                ent_coef=self.ent_coef,
                env=self.train_vec_env,
                n_epochs=self.n_epochs,
                tensorboard_log=f"./results/tensorboard/{self.results_folder_name}",
                verbose=1,
                # stats_window_size=self.n_steps / self.epi_length, # compute rollout statistics over the last iteration,
                device=self.device
            )
        else:
            import os
            folder_path, model_file = os.path.split(self.load_model_path)
            model_name = os.path.splitext(model_file)[0]
            prefix, num_timesteps, _ = model_name.split("_")
            env_path = os.path.join(folder_path, f"{prefix}_vecnormalize_{num_timesteps}_steps.pkl")
            # self.normed_vec_env = VecNormalize.load(env_path, self.dummy_vec_env)

            self.model = PPO.load(
                self.load_model_path, 
                env=self.train_vec_env,
                device=self.device,
                learning_rate=self.learning_rate,
                )
    


@hydra.main(version_base=None, config_path="psyonic_playing_xylophone/conf/thumb_wrist", config_name="train")
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