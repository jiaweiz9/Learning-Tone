import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from numpy.typing import ArrayLike
import torch
from typing import Dict, Any, Tuple
from psyonic_playing_xylophone.utils.sound_recorder import SoundRecorder
from psyonic_playing_xylophone.ros_interfaces.publisher import QPosPublisher, PAPRASJoint6PosPublisher
import rospy
import time, wandb, os
from psyonic_playing_xylophone.utils.reward_functions import RecRefRewardFunction
from psyonic_playing_xylophone.utils.reward_callback import RewardUtils
import librosa
import prettytable as pt
from sensor_msgs.msg import JointState
# from psyonic_hand_control.msg import handVal

class WristDemoEnv(gym.Env):

    metadata = {"render_modes": [None]}

    def __init__(self, single=True, render_mode=None):
        super().__init__()
        # self.config = config
        # action space: 0 := -10, 1 := no change, 2 := +10
        # self.action_space = gym.spaces.MultiDiscrete([6, 5])
        self.action_space = gym.spaces.MultiDiscrete([6, 5])

        self.observation_space = gym.spaces.Dict({
            # 'time_embedding': gym.spaces.Box(low=-1, high=1, shape=(2,)),
            'current_thumb_joint': gym.spaces.Box(low=-2, high=2, shape=(1,)),
            'previous_thumb_joint': gym.spaces.Box(low=-2, high=2, shape=(1,)),
            'current_wrist_joint': gym.spaces.Box(low=-2, high=2, shape=(1,)),
            'previous_wrist_joint': gym.spaces.Box(low=-2, high=2, shape=(1,)),
        })
        #self.state = np.zeros(5)
        #self.target = np.random.uniform(-1, 1, (5,))
        self.iteration = 0

        ####### amp06
        if single:
            self._action_to_thumb_movement = {
                0: -20,
                1: -10,
                2: -5,
                3: 0,
                4: 10,
                5: 10,
            }
        else:
            self._action_to_thumb_movement = {
            0: -20,
            1: -10,
            2: 0,
            3: 5,
            4: 10,
            5: 20
            }
            
        self._action_to_wrist_movement = {
            0: -0.075,
            1: -0.0375,
            2: 0,
            3: 0.0375,
            4: 0.075
        }
        

        self.initial_thumb_state = [50, 70 , 110, 115, 50, -10]
        self.initial_wrist_state = [0.1]
        if isinstance(self.initial_thumb_state, np.ndarray) is False or isinstance(self.initial_wrist_state, np.ndarray) is False:
            self.initial_thumb_state = np.array(self.initial_thumb_state)
            self.initial_wrist_state = np.array(self.initial_wrist_state)

        self.thumb_min_degree = -40
        self.thumb_max_degree = -10
        self.wrist_min_degree = -0.07
        self.wrist_max_degree = 0.15

        self.time_step = 0
        self.current_thumb_joint = self.initial_thumb_state[-1]
        self.previous_thumb_joint = self.initial_thumb_state[-1]
        self.current_wrist_joint = self.initial_wrist_state[-1]
        self.previous_wrist_joint = self.initial_wrist_state[-1]

        self.move_distance_curr_epi = 0
        self.epi_length = 50
        self.rollouts = 0

        self.__setup_command_publisher()
        # self.__load_reference_audio()
        
    def set_iteration(self, new_iter):
        self.iteration = new_iter

    def __time_step_embedding(self) -> ArrayLike:
        return np.array([np.sin(2 * np.pi * self.time_step / self.epi_length), 
                         np.cos(2 * np.pi * self.time_step / self.epi_length)])
    
    def __norm_thumb_obs(self, joint):
        return 2 * (joint - self.thumb_min_degree) / (self.thumb_max_degree - self.thumb_min_degree) - 1
    
    def __norm_wrist_obs(self, joint):
        return 2 * (joint - self.wrist_min_degree) / (self.wrist_max_degree - self.wrist_min_degree) - 1

    def _get_observation(self):
        return {
            # 'time_embedding': self.__time_step_embedding(),
            'current_thumb_joint': self.__norm_thumb_obs(self.current_thumb_joint) + self.__time_step_embedding()[0],
            'previous_thumb_joint': self.__norm_thumb_obs(self.previous_thumb_joint) + self.__time_step_embedding()[1],
            'current_wrist_joint': self.__norm_wrist_obs(self.current_wrist_joint) + self.__time_step_embedding()[0],
            'previous_wrist_joint': self.__norm_wrist_obs(self.previous_wrist_joint) + self.__time_step_embedding()[1]
        }
    
    # def _get_real_position(self):
    #     self.current_wrist_joint = rospy.wait_for_message("/joint_states", JointState, 0.5)
    #     self.current_thumb_joint = rospy.wait_for_message("/robot1/psyonic_hand_vals", handVal, 0.5)

    
    # def __load_reference_audio(self):
    #     if self.config["reference_audio"] is None:
    #         ref_audio_path = "ref_audio/ref_high.wav"
    #     else:
    #         ref_audio_path = self.config["reference_audio"]
        
    #     self.ref_audio, sr = librosa.load(ref_audio_path, sr=44100)
    #     self.ref_audio = np.pad(self.ref_audio[:88200], (0, 132300 - len(self.ref_audio)), 'constant')
    
    def __setup_command_publisher(self):
        rospy.init_node('psyonic_control', anonymous=True)
        # self.thumb_pos_publisher = QPosPublisher()
        self.wrist_pos_publisher = PAPRASJoint6PosPublisher()


    # def __episode_end_reward(self):
    #     if isinstance(self.last_rec_audio, np.ndarray) is False:
    #         raise ValueError("No valid recorded audio data found")

    #     # print(self.iteration)

    #     rec_ref_reward = RecRefRewardFunction(
    #         rec_audio=self.last_rec_audio,
    #         ref_audio=self.ref_audio,
    #         episode_length=self.config["epi_length"],
    #         sr=44100,
    #         iteration=self.iteration
    #     )

    #     # Set as attributes for logger to record
    #     self.last_amplitude_reward = rec_ref_reward.amplitude_reward()
    #     self.last_hitting_times_reward = rec_ref_reward.hitting_times_reward()
    #     self.last_onset_shape_reward = rec_ref_reward.onset_shape_reward()
    #     self.last_hitting_timing_reward = rec_ref_reward.hitting_timing_reward()
    #     self.success_reward = rec_ref_reward.success_reward()

    #     print(f"episode reward computed done!")
    #     return {
    #         "amplitude": self.last_amplitude_reward,
    #         "hitting_times": self.last_hitting_times_reward,
    #         "onset_shape": self.last_onset_shape_reward,
    #         "hitting_timing": self.last_hitting_timing_reward,
    #         "success":self.success_reward
    #     }
    
    def __ref_chunks_mean_nearby(self, rec_chunk_idx, before, after):
        # find the reference audio chunks in the range [rec_chunk_idx - before, rec_chunk_idx + after]
        start_idx = max(rec_chunk_idx - before, 0)
        end_idx = min(rec_chunk_idx + after, len(self.ref_audio) // 882 - 1)

        return self.ref_audio[start_idx * 882 : end_idx * 882]


    def step(self, action)-> Tuple[Dict[str, Any], int, bool, bool, dict]:
        self.time_step += 1

        if self.time_step == 1:
            # self.sound_recorder.start_recording()
            time.sleep(0.1)
            self.epi_start_time = time.time()
            self.last_chunk_idx=[]
            self.step_rewards=[]
            self.num_thumb_min_step = 0
            prev_step_rec_audio = 0

        self.previous_thumb_joint = self.current_thumb_joint
        self.previous_wrist_joint = self.current_wrist_joint
        self.current_thumb_joint += self._action_to_thumb_movement[action[0]]
        # self.current_thumb_joint = self._action_to_thumb_movement[action[0]]
        self.current_wrist_joint += self._action_to_wrist_movement[action[1]]

        # Clip the thumb and wrist joint command to make it within the feasible range
        self.current_thumb_joint = np.clip(self.current_thumb_joint,
                                           self.thumb_min_degree, 
                                           self.thumb_max_degree)
        self.move_distance_curr_epi += abs(self.current_thumb_joint - self.previous_thumb_joint)

        self.current_wrist_joint = np.clip(self.current_wrist_joint,
                                           self.wrist_min_degree,
                                           self.wrist_max_degree)
        self.move_distance_curr_epi += abs(self.current_wrist_joint - self.previous_wrist_joint) * 100


        next_thumb_movement, next_wrist_movement = self.get_state()
        # self.thumb_pos_publisher.publish_once(next_thumb_movement)
        self.wrist_pos_publisher.publish_once(next_thumb_movement, next_wrist_movement)

        # curr_step_rec_audio, chunk_index = self.sound_recorder.get_last_step_audio()
        # curr_step_ref_audio = self.ref_audio[chunk_index * 882 : (chunk_index + 1) * 882]

        # self.last_chunk_idx.append(chunk_index)

        terminated = True if self.time_step >= self.epi_length else False
        if terminated:
            # if self.config.get("short_epi", False):
            #     epi_duration = time.time() - self.epi_start_time
            #     print(epi_duration)
            #     time.sleep(2.3 - epi_duration)
            # self.sound_recorder.stop_recording()
            # audio_data = self.sound_recorder.get_episode_audio().squeeze()[4410:]
            
            # Early reset
            # self.wrist_pos_publisher.publish_once(np.concatenate(
            #     (self.initial_thumb_state[:-1], [self.thumb_min_degree]), axis=0), self.initial_wrist_state)
            self.wrist_pos_publisher.publish_once(self.initial_thumb_state, self.initial_wrist_state)

            #self.sound_recorder.save_recording()
            # data_fft = np.fft.fft(audio_data)
            # freqs = np.fft.fftfreq(len(data_fft), 1 / 44100)
            # data_fft[np.abs(freqs) < 1000] = 0
            # self.last_rec_audio = np.real(np.fft.ifft(data_fft))
            # print(f"Current episode data length: {audio_data.shape}")
            # self.sound_recorder.clear_buffer()
            # self.last_chunk_idx = np.array(self.last_chunk_idx)
            # self.last_chunk_idx = self.last_chunk_idx - self.last_chunk_idx[0]

        observation = self._get_observation()

        # Calculate rewards
        # 1. reward for thumb current step amplitude
        # reward = -self.config["reward_weight"]["amplitude_step"] * abs(np.mean(np.abs(curr_step_rec_audio)) - np.mean(np.abs(curr_step_ref_audio)))
        
        # self.step_rewards.append(reward.copy() / (self.config["reward_weight"]["amplitude_step"]))
        
        # 2. reward for thumb reducing shaking
        # reward += -(abs(self.current_thumb_joint - self.previous_thumb_joint) + 
        #             abs(self.current_wrist_joint - self.previous_wrist_joint) * 200) \
        #             * self.config["reward_weight"]["movement"]
        
        # if self.previous_thumb_joint == self.thumb_min_degree and self.current_thumb_joint == self.thumb_min_degree:
        #     reward += -self.config["reward_weight"]["movement"] * 2
        self.num_thumb_min_step += 1 if self.current_thumb_joint == self.thumb_min_degree else 0

        # if terminated:
        #     # 3. reward for playing the xylophone based on the recorded audio and the reference audio (only added at the end of the episode)
        #     table = pt.PrettyTable()
        #     for k, v in self.__episode_end_reward().items():
        #         if k == "success":
        #             success_reward = v      # 100 when success (over some thresholds), elso 0
        #             continue
        #         elif k == "amplitude":
        #             amplitude_reward = v
        #         elif k == "hitting_timing":
        #             timing_reward = v
        #         table.add_row([k, v])
                
        #         reward += self.config["reward_weight"][k] * v
            
        #     # 4. reward for good enough sound
        #     success_reward *= amplitude_reward * timing_reward * (1200 - self.move_distance_curr_epi) / 1200
        #     reward += success_reward
        #     table.add_row(["success reward", success_reward])

        #     # 5. reward for finger moving back to initial state
        #     moving_back_reward = 20 if self.current_thumb_joint >= self.initial_thumb_state[-1] \
        #                             and self.current_wrist_joint >= self.initial_thumb_state[-1] else 0
        #     reward += moving_back_reward

        #     reward += -self.num_thumb_min_step

        #     table.add_row(["Moving back", moving_back_reward])
        #     table.add_row(["Num of Min (Thumb)", -self.num_thumb_min_step])
        #     table.add_row(["Episode moving distance", self.move_distance_curr_epi])
        #     print(table)
        #     # self.rollouts += 1
            

        return observation, 0, terminated, False, {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # print(self.time_step)
        time.sleep(0.5)
        self.time_step = 0
        self.move_distance_curr_epi = 0

        # self.thumb_pos_publisher.publish_once(self.initial_thumb_state)
        self.current_thumb_joint = self.initial_thumb_state[-1]
        self.current_wrist_joint = self.initial_wrist_state[-1]
        print("Initial thumb joint: ", self.current_thumb_joint)
        print("Initial wrist joint: ", self.current_wrist_joint)
        print("iteration", self.iteration)
        # self.sound_recorder.start_recording()

        self.previous_thumb_joint = self.current_thumb_joint
        self.previous_wrist_joint = self.current_wrist_joint
        # self.last_chunk_idx = []
        # self.last_rec_audio = None
        # self.target = np.random.uniform(-1, 1, (5,))
        # print(self._get_observation())
        return self._get_observation(), {}


    def get_state(self) -> ArrayLike:
        # using current command as joint state (Note: this is not the actual joint state)
        return np.concatenate([self.initial_thumb_state[:-1], [self.current_thumb_joint]], axis=0), \
                np.concatenate([self.initial_wrist_state[:-1], [self.current_wrist_joint]], axis=0)
    
    def close(self):
        # del self.sound_recorder
        # del self.thumb_pos_publisher
        del self.wrist_pos_publisher
        rospy.signal_shutdown("Shutting down the node")

        return super().close()