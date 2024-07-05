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

class PsyonicThumbDoubleEnv(gym.Env):

    metadata = {"render_modes": [None]}

    def __init__(self, config: Dict[str, Any], render_mode=None):
        super().__init__()
        self.config = config
        # action space: 0 := -10, 1 := no change, 2 := +10
        self.action_space = gym.spaces.Discrete(6)

        self.observation_space = gym.spaces.Dict({
            # 'time_embedding': gym.spaces.Box(low=-1, high=1, shape=(2,)),
            'current_thumb_joint': gym.spaces.Box(low=-2, high=2, shape=(1,)),
            'previous_thumb_joint': gym.spaces.Box(low=-2, high=2, shape=(1,))
        })
        #self.state = np.zeros(5)
        #self.target = np.random.uniform(-1, 1, (5,))
        self.iteration = 0

        ####### amp06
        self._action_to_joint_movement = {
            0: -20,
            1: -10,
            2: 0,
            3: 5,
            4: 10,
            5: 20
        }
        # self._action_to_wrist_movement = {
        #     0: -0.075,
        #     1: -0.0375,
        #     2: -0.0185,
        #     3: 0,
        #     4: 0.0185,
        #     5: 0.0375,
        #     6: 0.075
        # }
        
        ####### amp043
        # self._action_to_thumb_movement = {
        #     0: -20,
        #     1: -10,
        #     2: -5,
        #     3: 0,
        #     4: 5,
        #     5: 10,
        #     6: 20,
        # }

        # self._action_to_wrist_movement = {
        #     0: -0.075,
        #     1: -0.0375,
        #     2: -0.01875,
        #     3: 0,
        #     4: 0.01875,
        #     5: 0.0375,
        #     6: 0.075
        # }

        self.initial_joints_state = config["psyonic"]["initial_state"]
        if isinstance(self.initial_joints_state, np.ndarray) is False:
            self.initial_joints_state = np.array(self.initial_joints_state)

        self.min_degree = config["psyonic"]["min_degree"]
        self.max_degree = config["psyonic"]["max_degree"]
        # self.ref_audio_path = config["reference_audio"]

        self.time_step = 0
        self.current_thumb_joint = self.initial_joints_state[-1]
        self.previous_thumb_joint = self.current_thumb_joint

        self.sound_recorder = SoundRecorder()
        self.reward = 0
        self.last_rec_audio = None
        self.last_chunk_idx = []
        self.step_rewards = []
        self.move_rew_weight = config["reward_weight"]["movement"]
        # self.amp_step_rew_weight = config["reward_weight"]["amplitude_step"]
        self.move_distance_curr_epi = 0
        self.epi_length = self.config["epi_length"]
        self.rollouts = 0

        self.__setup_command_publisher()
        self.__load_reference_audio()
        # self.reward_utils = RewardUtils(
        #     self.ref_audio,
        #     episode_length=50,
        #     )
        self.firstInRange1 = True
        self.firstInRange2 = True

    def set_iteration(self, new_iter):
        self.iteration = new_iter

    def __time_step_embedding(self) -> ArrayLike:
        return np.array([np.sin(2 * np.pi * self.time_step / self.epi_length), 
                         np.cos(2 * np.pi * self.time_step / self.epi_length)])
    
    def __norm_obs(self, joint):
        return 2 * (joint - self.min_degree) / (self.max_degree - self.min_degree) - 1

    def _get_observation(self):
        return {
            # 'time_embedding': self.__time_step_embedding(),
            'current_thumb_joint': self.__norm_obs(self.current_thumb_joint) + self.__time_step_embedding()[0],
            'previous_thumb_joint': self.__norm_obs(self.previous_thumb_joint) + self.__time_step_embedding()[1],
        }
    # def _get_real_position(self):
    #     self.current_wrist_joint = rospy.wait_for_message("/joint_states", JointState, 0.5)
    #     self.current_thumb_joint = rospy.wait_for_message("/robot1/psyonic_hand_vals", handVal, 0.5)

    
    def __load_reference_audio(self):
        if self.config["reference_audio"] is None:
            ref_audio_path = "ref_audio/ref_high.wav"
        else:
            ref_audio_path = self.config["reference_audio"]
        
        self.ref_audio, sr = librosa.load(ref_audio_path, sr=44100)
        self.ref_audio = np.pad(self.ref_audio[:88200], (0, 132300 - len(self.ref_audio)), 'constant')

        self._ref_onset_strength_envelop = librosa.onset.onset_strength(y=self.ref_audio, sr=sr)
        self._ref_onset_strength_envelop[self._ref_onset_strength_envelop < 10] = 0
        self._ref_hitting_timings = librosa.onset.onset_detect(onset_envelope=self._ref_onset_strength_envelop, sr=sr, units='time', normalize=True)
        self._ref_hitting_frames = (self._ref_hitting_timings * sr).astype(int)


    def isTimingWithinRange(self, rec_audio_chunk_idx, range=5):
        for idx, ref_hitting_frame in enumerate(self._ref_hitting_frames):
            if ref_hitting_frame//882 - range <= rec_audio_chunk_idx <= ref_hitting_frame//882 + range:
                print("Hitting Within Range!")
                print(f"rec hitting chunk idx {rec_audio_chunk_idx}, ref hitting chunk idx {ref_hitting_frame}")
                return True, idx
        return False, -1
    
    def __setup_command_publisher(self):
        rospy.init_node('psyonic_control', anonymous=True)
        self.qpos_publisher = QPosPublisher()


    def __episode_end_reward(self):
        if isinstance(self.last_rec_audio, np.ndarray) is False:
            raise ValueError("No valid recorded audio data found")

        # print(self.iteration)

        rec_ref_reward = RecRefRewardFunction(
            rec_audio=self.last_rec_audio,
            ref_audio=self.ref_audio,
            episode_length=self.config["epi_length"],
            sr=44100,
            iteration=self.iteration
        )

        # Set as attributes for logger to record
        self.last_amplitude_reward = rec_ref_reward.double_hit_amp_reward()
        self.last_hitting_times_reward = rec_ref_reward.hitting_times_reward()
        self.last_onset_shape_reward = rec_ref_reward.onset_shape_reward()
        self.last_hitting_timing_reward = rec_ref_reward.hitting_timing_reward()
        self.success_reward = rec_ref_reward.double_hit_success_reward()

        print(f"episode reward computed done!")
        return {
            "amplitude": self.last_amplitude_reward,
            "hitting_times": self.last_hitting_times_reward,
            "onset_shape": self.last_onset_shape_reward,
            "hitting_timing": self.last_hitting_timing_reward,
            "success":self.success_reward
        }
    
    def __ref_chunks_mean_nearby(self, rec_chunk_idx, before, after):
        # find the reference audio chunks in the range [rec_chunk_idx - before, rec_chunk_idx + after]
        start_idx = max(rec_chunk_idx - before, 0)
        end_idx = min(rec_chunk_idx + after, len(self.ref_audio) // 882 - 1)

        return self.ref_audio[start_idx * 882 : end_idx * 882]


    def step(self, action)-> Tuple[Dict[str, Any], int, bool, bool, dict]:
        self.time_step += 1

        if self.time_step == 1:
            self.sound_recorder.start_recording()
            time.sleep(0.1)
            self.epi_start_time = time.time()
            self.last_chunk_idx=[]
            self.step_rewards=[]
            self.num_thumb_min_step = 0
            self.prev_step_rec_audio = 0
            self.isFirstHitting = True

        self.previous_thumb_joint = self.current_thumb_joint
        self.current_thumb_joint += self._action_to_joint_movement[action]

        # Clip the thumb and wrist joint command to make it within the feasible range
        self.current_thumb_joint = np.clip(self.current_thumb_joint,
                                           self.min_degree, 
                                           self.max_degree)
        self.move_distance_curr_epi += abs(self.current_thumb_joint - self.previous_thumb_joint)


        next_movement = self.get_state()
        # self.thumb_pos_publisher.publish_once(next_thumb_movement)
        self.qpos_publisher.publish_once(next_movement)

        curr_step_rec_audio, chunk_index = self.sound_recorder.get_last_step_audio()


        if self.time_step == 1:
            self.first_chunk_idx = chunk_index
        chunk_index = chunk_index - self.first_chunk_idx
        curr_step_ref_audio = self.ref_audio[chunk_index * 882 : (chunk_index + 1) * 882]

        self.last_chunk_idx.append(chunk_index)

        terminated = True if self.time_step >= self.config["epi_length"] else False
        if terminated:
            if self.config.get("short_epi", False):
                epi_duration = time.time() - self.epi_start_time
                print(epi_duration)
                time.sleep(2.3 - epi_duration)
            self.sound_recorder.stop_recording()
            audio_data = self.sound_recorder.get_episode_audio().squeeze()[4410:]

            #self.sound_recorder.save_recording()
            data_fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(data_fft), 1 / 44100)
            data_fft[np.abs(freqs) < 1000] = 0
            self.last_rec_audio = np.real(np.fft.ifft(data_fft))
            print(f"Current episode data length: {audio_data.shape}")
            self.sound_recorder.clear_buffer()
            self.last_chunk_idx = np.array(self.last_chunk_idx)
            self.last_chunk_idx = self.last_chunk_idx - self.last_chunk_idx[0]

        observation = self._get_observation()

        # Calculate rewards
        # 1. reward for thumb current step amplitude
        # reward = -self.config["reward_weight"]["amplitude_step"] * abs(np.mean(np.abs(curr_step_rec_audio)) - np.mean(np.abs(curr_step_ref_audio)))
        # isWithinRange, idx = self.isTimingWithinRange(rec_audio_chunk_idx=chunk_index, range=10)

        # if np.max(curr_step_rec_audio) > self.prev_step_rec_audio + 0.1 and isWithinRange and ((self.firstInRange1 and idx == 0) or (self.firstInRange2 and idx == 1)):
        #     reward = (-abs(self._ref_hitting_frames[idx] // 882 - chunk_index) / 20 + 1) * np.array(5.0) * self.config["reward_weight"]["amplitude_step"]
        #     if self.firstInRange1 and idx == 0:
        #         self.firstInRange1 = False
        #     if self.firstInRange2 and idx == 1:
        #         self.firstInRange2 == False
        #     print(f"Hitting on time reward {reward}")
        #     # reward = np.array(5.0) * self.config["reward_weight"]["amplitude_step"]
        #     # self.isFirstHitting = 
        # elif np.max(curr_step_rec_audio) > self.prev_step_rec_audio + 0.1 and not isWithinRange:
        #     reward = -np.array(2.0) * self.config["reward_weight"]["amplitude_step"]
        # else:
        #     reward = -self.config["reward_weight"]["amplitude_step"] * abs(np.mean(np.abs(curr_step_rec_audio)) - np.mean(np.abs(curr_step_ref_audio)))


        if np.max(curr_step_rec_audio) > self.prev_step_rec_audio + 0.1:
            isWithinRange, idx = self.isTimingWithinRange(rec_audio_chunk_idx=chunk_index, range=10)
            if isWithinRange and ((self.firstInRange1 and idx == 0) or (self.firstInRange2 and idx == 1)):
                reward = (-abs(self._ref_hitting_frames[idx] // 882 - chunk_index) / 20 + 1) * np.array(5.0) * self.config["reward_weight"]["amplitude_step"]
                if self.firstInRange1 and idx == 0:
                    self.firstInRange1 = False
                if self.firstInRange2 and idx == 1:
                    self.firstInRange2 = False
                print(f"Hitting on time reward {reward}")
                # reward = np.array(5.0) * self.config["reward_weight"]["amplitude_step"]
                # self.isFirstHitting = 
            elif not isWithinRange:
                reward = -np.array(2.0) * self.config["reward_weight"]["amplitude_step"]
            else:
               reward = -self.config["reward_weight"]["amplitude_step"] * abs(np.mean(np.abs(curr_step_rec_audio)) - np.mean(np.abs(curr_step_ref_audio)))
 
        else:
            reward = -self.config["reward_weight"]["amplitude_step"] * abs(np.mean(np.abs(curr_step_rec_audio)) - np.mean(np.abs(curr_step_ref_audio)))



        self.prev_step_rec_audio = np.max(curr_step_rec_audio)

        self.step_rewards.append(reward.copy() / (self.config["reward_weight"]["amplitude_step"]))
        
        # 2. reward for thumb reducing shaking
        reward += -abs(self.current_thumb_joint - self.previous_thumb_joint) * self.config["reward_weight"]["movement"]
        
        # if self.previous_thumb_joint == self.thumb_min_degree and self.current_thumb_joint == self.thumb_min_degree:
        #     reward += -self.config["reward_weight"]["movement"] * 2
        # self.num_thumb_min_step += 1 if self.current_thumb_joint == self.thumb_min_degree else 0

        if terminated:
            # 3. reward for playing the xylophone based on the recorded audio and the reference audio (only added at the end of the episode)
            table = pt.PrettyTable()
            for k, v in self.__episode_end_reward().items():
                if k == "success":
                    success_reward = v      # 100 when success (over some thresholds), elso 0
                    continue
                elif k == "amplitude":
                    amplitude_reward = v
                elif k == "hitting_timing":
                    timing_reward = v
                table.add_row([k, v])
                
                reward += self.config["reward_weight"][k] * v
            
            # 4. reward for good enough sound
            amp_coefficient =(amplitude_reward - 0.4) / 6 + 0.9
            success_reward *= amp_coefficient * timing_reward * (1000 - self.move_distance_curr_epi) / 1000
            reward += success_reward
            table.add_row(["success reward", success_reward])

            # 5. reward for finger moving back to initial state
            moving_back_reward = 20 if self.current_thumb_joint >= self.initial_joints_state[-1] else 0

            reward += moving_back_reward

            # reward += -self.num_thumb_min_step

            table.add_row(["Moving back", moving_back_reward])
            # table.add_row(["Num of Min (Thumb)", -self.num_thumb_min_step])
            table.add_row(["Episode moving distance", self.move_distance_curr_epi])
            print(table)
            # self.rollouts += 1
            

        return observation, reward, terminated, False, {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # print(self.time_step)
        time.sleep(0.5)
        self.time_step = 0
        self.move_distance_curr_epi = 0

        # self.thumb_pos_publisher.publish_once(self.initial_thumb_state)
        self.current_thumb_joint = self.initial_joints_state[-1]
        self.qpos_publisher.publish_once(self.get_state())
        print("Initial thumb joint: ", self.current_thumb_joint)
        # print("Initial wrist joint: ", self.current_wrist_joint)
        time.sleep(0.5)
        # print("iteration", self.iteration)
        # self.sound_recorder.start_recording()

        self.previous_thumb_joint = self.current_thumb_joint
        # self.previous_wrist_joint = self.current_wrist_joint
        
        # self.last_chunk_idx = []
        # self.last_rec_audio = None
        # self.target = np.random.uniform(-1, 1, (5,))
        # print(self._get_observation())
        self.firstInRange1 = True
        self.firstInRange2 = True
        return self._get_observation(), {}


    def get_state(self) -> ArrayLike:
        # using current command as joint state (Note: this is not the actual joint state)
        return np.concatenate([self.initial_joints_state[:-1],
                               [self.current_thumb_joint]],
                               axis=0)
    
    def close(self):
        del self.sound_recorder
        del self.qpos_publisher
        rospy.signal_shutdown("Shutting down the node")

        return super().close()


if __name__ == "__main__":
    config = {
        "psyonic": {
            "initial_state": np.array([0, 0, 0, 0, 0]),
            "min_degree": -50,
            "max_degree": -10,
            },

        "papras_joint6": {
            "initial_state": [0],
            "max_degree": 10,
            "min_degree": 0,
            },

        "reward_weight": {
            "amplitude_step": 1,
            "movement": 0.005,
            "amplitude": 10,
            "hitting_times": 10,
            "onset_shape": 10,
            "hitting_timing": 40,
        },
        "epi_length": 50,
        "reference_audio": "ref_audio/xylophone_2s/amp03_clip.wav"
        }
    
    # print(os.path.realpath(__file__))
    env = PsyonicThumbDoubleEnv(config=config)
    env = gym.wrappers.FlattenObservation(env)

    obs, info = env.reset()
    print("Initial observation: ", obs)
    for _ in range(50):
        action = env.action_space.sample()
        env.step(action)
    env.close()
    print("done!")