import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

#TODO: Add VecNormalized environment saving process
class TestPPO:
    def __init__(self, config) -> None:
        self.env_id = config["env_id"]
        self.policy = config["policy"]
        
        self.load_model_path = config.get("load_model_path", None)
        self.epi_length = config["epi_length"]

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
            norm_reward=False, # No need to normalize reward when testing
            clip_obs=10.
            )
        
        self.__load_env()
        self.__load_model()
        
    def __load_env(self):
        self.dummy_vec_env.load("vec_normalize.pkl")
        self.dummy_vec_env.training = False


    def __load_model(self):
        self.model = PPO.load(self.load_model_path)
        self.model.set_env(self.dummy_vec_env)

    def do_predict(self):
        obs = self.dummy_vec_env.reset()
        for _ in range(self.epi_length):
            action, _ = self.model.predict(obs)
            obs, reward, done, info = self.dummy_vec_env.step(action)
            self.dummy_vec_env.render()
            if done:
                obs = self.dummy_vec_env.reset()
        self.dummy_vec_env.close()


@hydra.main(version_base=None, config_path="psyonic_playing_xylophone/conf/psyonic_thumb", config_name="test")
def launch_test(cfg: DictConfig) -> None:

    cfg = OmegaConf.to_container(cfg, resolve=True)
    test_ppo = TestPPO(cfg)

    test_ppo.do_predict()

if __name__ == "__main__":
    launch_test()