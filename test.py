import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import os
from psyonic_playing_xylophone.envs.psyonic_thumb import PsyonicThumbEnv
from psyonic_playing_xylophone.utils.vis_result import VisualizeEpisodeCallback
import matplotlib.pyplot as plt
import numpy as np

class TestPPO:
    def __init__(self, config) -> None:
        self.env_id = config["env_id"]
        self.policy = config["policy"]
        
        self.load_model_path = config.get("load_model_path", None)
        self.use_vecnorm = config.get("use_vecnorm", False)
        self.device = config.get("device", "auto")
        
        folder_path, model_file = os.path.split(self.load_model_path)
        model_name = os.path.splitext(model_file)[0]
        if self.use_vecnorm:
            prefix, num_timesteps, _ = model_name.split("_")
            self.load_env_path = os.path.join(folder_path, f"{prefix}_vecnormalize_{num_timesteps}_steps.pkl")

        self.epi_length = config["epi_length"]
        self.dummy_vec_env = make_vec_env(
            self.env_id, 
            n_envs=1, 
            seed=0, 
            wrapper_class=gym.wrappers.FlattenObservation, 
            env_kwargs={"config": config},
            )
        
        self.__load_env()
        self.__load_model()
        
    def __load_env(self):
        if self.use_vecnorm:
            self.normed_vec_env = VecNormalize.load(self.load_env_path, self.dummy_vec_env)
            self.normed_vec_env.training = False
            print(f"obs rms: mean {self.normed_vec_env.obs_rms.mean}, var {self.normed_vec_env.obs_rms.var}")
            print(f"ret rms: mean {self.normed_vec_env.ret_rms.mean}, var {self.normed_vec_env.ret_rms.var}")
        else:
            self.normed_vec_env = self.dummy_vec_env


    def __load_model(self):
        self.model = PPO.load(self.load_model_path, device=self.device)
        self.model.set_env(self.normed_vec_env) 

    def do_predict(self, iter=1):
        obs = self.normed_vec_env.reset()
        print(f"init obs {obs}")
        for i in range(iter):
            for _ in range(self.epi_length):
                action, _ = self.model.predict(obs, deterministic=True)
                print(action)
                obs, reward, done, info = self.normed_vec_env.step(action)
                if done:
                    obs = self.normed_vec_env.reset()
                    break
            self.rec_audio = self.normed_vec_env.get_attr("last_rec_audio")[0]
            self.ref_audio = self.normed_vec_env.get_attr("ref_audio")[0]
        import wavio
        wavio.write(f"predicted_audio.wav", self.rec_audio[:88200], rate=44100, sampwidth=4)
        self.normed_vec_env.close()

    def _visualize_audio_step(self, sr=44100) -> None:
        assert self.rec_audio is not None and self.ref_audio is not None, "No recorded data yet"
        plt.figure(figsize=(20, 6))
        max_len = max(len(self.rec_audio), len(self.ref_audio))
        self.ref_audio = np.pad(self.ref_audio, (0, max_len - len(self.ref_audio)), 'constant')
        self.rec_audio = np.pad(self.rec_audio, (0, max_len - len(self.rec_audio)), 'constant')

        time = np.arange(0, max_len) / sr

        plt.plot(time, self.rec_audio, color='blue', alpha=0.3)
        plt.plot(time, self.ref_audio, color='red', alpha=0.3)

        # rec_idx = rec_idx * 0.02
        # plt.scatter(rec_idx, step_rew, color='black', marker='x')

        # plt.title(f'Episode {rec_idx} - Reference Audio (red) vs. Recorded Audio (blue)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend(['Recorded Audio', 'Reference Audio'])

        plt.savefig("Predict")
        plt.close()


@hydra.main(version_base=None, config_path="psyonic_playing_xylophone/conf/thumb_double", config_name="test")
def launch_test(cfg: DictConfig) -> None:

    cfg = OmegaConf.to_container(cfg, resolve=True)
    test_ppo = TestPPO(cfg)

    test_ppo.do_predict(3)
    test_ppo._visualize_audio_step()

if __name__ == "__main__":
    launch_test()