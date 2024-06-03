import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from numpy.typing import ArrayLike
import torch
from typing import Dict, Any, Tuple
from psyonic_playing_xylophone.utils.sound_recorder import SoundRecorder
from psyonic_playing_xylophone.ros_interfaces.publisher import QPosPublisher
import rospy
import time, wandb, os
from psyonic_playing_xylophone.utils.reward_functions import RecRefRewardFunction
import librosa

class PsyonicThumbEnv(gym.Env):

    metadata = {"render_modes": [None]}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        # action space: 0 := -10, 1 := no change, 2 := +10
        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Dict({
            'time_embedding': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'current_thumb_joint': gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'previous_thumb_joint': gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })
        #self.state = np.zeros(5)
        #self.target = np.random.uniform(-1, 1, (5,))

        self._action_to_joint_movement = {
            0: -10,
            1: 0,
            2: 10
        }

        self.initial_joints_state = config["psyonic"]["initial_state"]
        if isinstance(self.initial_joints_state, np.ndarray) is False:
            self.initial_joints_state = np.array(self.initial_joints_state)

        self.min_degree = config["psyonic"]["min_degree"]
        self.max_degree = config["psyonic"]["max_degree"]
        # self.ref_audio_path = config["reference_audio"]

        self.time_step = 0
        self.current_thumb_joint = 0
        self.previous_thumb_joint = 0
        self.sound_recorder = SoundRecorder()
        self.reward = 0
        self.last_rec_audio = None

        self.__setup_command_publisher()
        self.__load_reference_audio()

    def __time_step_embedding(self) -> ArrayLike:
        return np.array([np.sin(self.time_step), 
                         np.cos(self.time_step)])

    def _get_observation(self):
        return {
            'time_embedding': self.__time_step_embedding(),
            'current_thumb_joint': self.current_thumb_joint,
            'previous_thumb_joint': self.previous_thumb_joint
        }
    
    def __load_reference_audio(self):
        if self.config["reference_audio"] is None:
            ref_audio_path = "ref_audio/ref_high.wav"
        else:
            ref_audio_path = self.config["reference_audio"]
        
        self.ref_audio, sr = librosa.load(ref_audio_path, sr=44100)
    
    def __setup_command_publisher(self):
        rospy.init_node('psyonic_control', anonymous=True)
        self.qpos_publisher = QPosPublisher()

    def __episode_end_reward(self):
        if isinstance(self.last_rec_audio, np.ndarray) is False:
            raise ValueError("No valid recorded audio data found")

        rec_ref_reward = RecRefRewardFunction(
            rec_audio=self.last_rec_audio,
            ref_audio=self.ref_audio,
            episode_length=self.config["epi_length"],
            sr=44100
        )

        return (
            self.config["reward_weight"]["amplitude"] * rec_ref_reward.amplitude_reward() +
            self.config["reward_weight"]["hitting_times"] * rec_ref_reward.hitting_times_reward() +
            self.config["reward_weight"]["onset_shape"] * rec_ref_reward.onset_shape_reward() +
            self.config["reward_weight"]["hitting_timing"] * rec_ref_reward.hitting_timing_reward() 
        )

    def step(self, action)-> Tuple[Dict[str, Any], int, bool, bool, dict]:
        self.time_step += 1
        self.previous_thumb_joint = self.current_thumb_joint
        self.current_thumb_joint += self._action_to_joint_movement[action]
        self.current_thumb_joint = np.clip(self.current_thumb_joint,
                                           self.min_degree, 
                                           self.max_degree)

        next_movement = self.get_state()
        self.qpos_publisher.publish_once(next_movement)

        terminated = True if self.time_step >= self.config["epi_length"] else False
        if terminated:
            self.sound_recorder.stop_recording()
            audio_data = self.sound_recorder.get_current_buffer()
            #self.sound_recorder.save_recording()
            self.sound_recorder.clear_buffer()
            self.last_rec_audio = audio_data

        # Calculate rewards
        # 1. reward for moving thumb joint not too fast
        reward = -1 if abs(self.current_thumb_joint - self.previous_thumb_joint) > 10 else 0
        # 2. reward for playing the xylophone based on the recorded audio and the reference audio (only added at the end of the episode)
        if terminated:
            reward += self.__episode_end_reward()
            
        observation = self._get_observation()
        print(f"Step: {self.time_step}, Reward: {reward}", "Observation: ", observation)

        return observation, reward, terminated, False, {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.time_step = 0

        self.current_thumb_joint = self.initial_joints_state[-1]
        self.qpos_publisher.publish_once(self.initial_joints_state)
        print("Initial thumb joint: ", self.current_thumb_joint)


        self.sound_recorder.start_recording()

        self.previous_thumb_joint = self.current_thumb_joint
        # self.target = np.random.uniform(-1, 1, (5,))
        return self._get_observation(), {}


    def get_state(self) -> ArrayLike:
        # using previous command as joint state (Note: this is not the actual joint state)
        return np.concatenate([self.initial_joints_state[:-1],
                               [self.current_thumb_joint]],
                               axis=0)


if __name__ == "__main__":
    config = {
        "psyonic": {
            "initial_state": np.array([0, 0, 0, 0, 0]),
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
    
    # print(os.path.realpath(__file__))
    env = PsyonicThumbEnv(config=config)
    env = gym.wrappers.FlattenObservation(env)

    obs, info = env.reset()
    print("Initial observation: ", obs)
    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)
    env.close()
    print("done!")