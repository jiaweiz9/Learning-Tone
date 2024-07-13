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
import rospy
from std_msgs.msg import Float32MultiArray, Float64MultiArray, Bool

class WristDemo:
    def __init__(self) -> None:
        self.env_id = "WristDemoEnv-v0"
        
        self.load_single_hit_path = "results/model_demo/single_hit_1.zip"

        self.load_double_hit_path = "results/model_demo/double_hit.zip"

        self.epi_length = 50
        self.single_hit_env = make_vec_env(
            self.env_id, 
            n_envs=1, 
            seed=0, 
            wrapper_class=gym.wrappers.FlattenObservation, 
            env_kwargs={"single": True},
        )
        
        self.double_hit_env = make_vec_env(
            self.env_id, 
            n_envs=1,
            seed=0, 
            wrapper_class=gym.wrappers.FlattenObservation, 
            env_kwargs={"single": False},
        )
        
        # self.__load_env()
        # self.__load_model()
        self.single_hit_model = PPO.load(self.load_single_hit_path)
        self.single_hit_model.set_env(self.single_hit_env)

        self.double_hit_model = PPO.load(self.load_double_hit_path)
        self.double_hit_model.set_env(self.double_hit_env)
        
    def init_listener(self):
        # rospy.init_node('demo_listener', anonymous=True)
        print("init subscriber")
        rospy.Subscriber("xylophone_demo/single_hit", Bool, self.do_predict_single_hit)
        rospy.Subscriber("xylophone_demo/double_hit", Bool, self.do_predict_double_hit)

        rospy.spin()


    def do_predict_single_hit(self, message):
        print(message)

        if message.data == True:
            print("Do predict single hit")

            obs = self.single_hit_env.reset()
            
            for _ in range(self.epi_length):
                action, _ = self.single_hit_model.predict(obs, deterministic=True)
                print(action)
                obs, reward, done, info = self.single_hit_env.step(action)
                if done:
                    obs = self.single_hit_env.reset()
                    break
            # self.rec_audio = self.normed_vec_env.get_attr("last_rec_audio")[0]
            # self.ref_audio = self.normed_vec_env.get_attr("ref_audio")[0]
        # import wavio
        # wavio.write(f"predicted_audio.wav", self.rec_audio[:88200], rate=44100, sampwidth=4)
        # self.single_hit_env.close()

    def do_predict_double_hit(self, message):
        print(message)
        if message.data == True:
            print("Do predict single hit")
            obs = self.double_hit_env.reset()
            # print(f"init obs {obs}")
            for _ in range(self.epi_length):
                action, _ = self.double_hit_model.predict(obs, deterministic=True)
                print(action)
                obs, reward, done, info = self.double_hit_env.step(action)
                if done:
                    obs = self.double_hit_env.reset()
                    break
            # self.rec_audio = self.normed_vec_env.get_attr("last_rec_audio")[0]
            # self.ref_audio = self.normed_vec_env.get_attr("ref_audio")[0]
        # import wavio
        # wavio.write(f"predicted_audio.wav", self.rec_audio[:88200], rate=44100, sampwidth=4)
        # self.double_hit_env.close()

if __name__ == "__main__":
    demo = WristDemo()
    demo.init_listener()
    # demo.do_predict_single_hit()
    # demo.do_predict_double_hit()
