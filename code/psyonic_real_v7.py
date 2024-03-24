import wandb, os, copy, sys, rospy, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.psyonic_func import *
from rl.ppo import *
from ros_func.publisher import QPosPublisher
import numpy as np
import librosa
import wavio
from utils.sound_recorder import SoundRecorder
from utils.reward_functions import amplitude_reward, onset_rewards
from utils.psyonic_func import get_velocity, vel_clip_action, get_acceleration, beta_dist_to_action_space
from utils.logger import Logger


class PsyonicForReal():
    def __init__(self, args):

        # reward setting
        self.w_amp_rew = 1
        self.w_onset_rew = 1e2
        self.w_timing_rew = 1e2
        self.w_hit_rew = 1e2

        # recorder setting
        self.ref_audio_path = args.ref_audio_path
        self.record_duration = args.record_duration
        self.ros_rate = args.ros_rate
        self.samplerate = args.samplerate

        # training setting
        self.n_epi = args.n_epi
        self.k_epoch = args.k_epoch
        self.mini_batch_size = args.mini_batch_size
        self.max_iter = args.max_iter
        self.weight_iter_num = args.weight_iter_num
        self.SAVE_WEIGHTS = args.SAVE_WEIGHTS
        self.seed = args.seed

        # action setting
        self.initial_pose = np.array([105, 105, 105, 110, 70, -10])
        self.pose_upper = np.array([105, 105, 105, 110, 70, -10])
        self.pose_lower = np.array([95, 95, 95, 100, 65, -35])
        print("initial pose:", self.initial_pose)
        self.velocity_free_coef = args.velocity_free_coef
        self.min_vel = args.min_vel
        self.max_vel = args.max_vel

        # PPO setting
        self.obs_dim = args.obs_dim
        self.act_dim = args.act_dim
        self.h_dims = args.h_dims
        self.gamma = args.gamma
        self.lmbda = args.lmbda
        self.lr_actorcritic = args.lr_actorcritic
        self.clip_ratio = args.clip_ratio
        self.value_coef = args.value_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad = args.max_grad
        self.beta_dist = args.beta_dist

        # logger setting
        log_config = {"w_amp_rew": self.w_amp_rew, "w_onset_rew": self.w_onset_rew, "w_timing_rew": self.w_timing_rew, "w_hit_rew": self.w_hit_rew,
                      "n_epi": self.n_epi, "k_epoch": self.k_epoch, "mini_batch_size": self.mini_batch_size, "intial_pose": self.initial_pose, "velocity_free_coef": self.velocity_free_coef,
                      "max_vel": self.max_vel, "obs_dim": self.obs_dim, "act_dim": self.act_dim, "beta_dist": self.beta_dist}

        self.logger = Logger(args.WANDB, log_config)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_ros()

    def set_random_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed) 
        torch.backends.cudnn.deterministic = True

    def set_ros(self):
        rospy.init_node('psyonic_for_real', anonymous=True)
        self.QPosPublisher = QPosPublisher()
    
    
    def sample_trajectory(self, PPO_agent, buffer, max_step, episode_len, min_vel, max_vel, samplerate):
        obs_trajectory = []
        act_trajectory = []
        reward_trajectory = []
        info = {}

        episode_reward = 0
        episode_rewards = []
        epi_onset_rewards = []
        epi_timing_rewards = []
        epi_hit_rewards = []
        
        for i in range(max_step):

            # For every new episode, we reset the position to the minimum value?, obersevation to zero
            if i % episode_len == 0:
                episode_reward = 0
                prev_action = self.initial_pose
                pre_velocity = np.zeros(6)
                velocity = np.zeros(6)
                acceleration = np.zeros(6)

                obs = np.concatenate((prev_action, self.initial_pose, np.zeros(19)), axis=0) # initial obersavation is initial pose with 0 velocity and 0 acceleration
                # obs_trajectory.append(obs)

                # Start recording
                Recoder = SoundRecorder(samplerate=samplerate, audio_device=None) # Bug fixed!! audio_devce=None is to use default connected device
                Recoder.start_recording()
                time.sleep(0.1)
                start_time = time.time()

            else:
                prev_action = act_trajectory[i-1]
            
            # Get action from actor
            action, log_prob, val = PPO_agent.get_action(obs)
            
            if self.beta_dist:
                curr_action = beta_dist_to_action_space(action, self.pose_lower, self.pose_upper)
            else:
                curr_action = np.clip(action, self.pose_lower, self.pose_upper)

            
            # Clip action based on velocity
            curr_action, vel_clip = vel_clip_action(prev_action, curr_action, min_vel=min_vel, max_vel=max_vel, ros_rate=self.ros_rate) # radian

            self.QPosPublisher.publish_once(curr_action) # Publish action 0.02 sec

            velocity = get_velocity(prev_action, curr_action, ros_rate=self.ros_rate)
            acceleration = get_acceleration(pre_velocity, velocity, ros_rate=self.ros_rate)

            # Get audio data
            cur_time = time.time()

            ref_audio, ref_sr = librosa.load(self.ref_audio_path, sr=44100) # load reference audio
            ref_audio_info = ref_audio[882* (i % episode_len) : 882* ((i  % episode_len) + 1)]

            audio_data = Recoder.get_current_buffer() # get current sound data
            ###############TODO: decide how to match#################
            audio_data_step = audio_data[882* (i % episode_len) : 882* ((i  % episode_len) + 1)] # time window slicing 0.02sec

            amp_reward, mean_amp = amplitude_reward(audio_data_step, ref_audio_info, amp_scale=1e2)

            # Set next observation
            mean_amp = np.asarray(mean_amp).reshape(1)
            next_obs = np.concatenate((prev_action, curr_action, pre_velocity, velocity, acceleration, mean_amp), axis=0)
            pre_velocity = velocity

            onset_reward = 0
            timing_reward = 0
            hit_reward = 0
            
            # Total Reward
            step_reward = self.w_amp_rew * amp_reward
            episode_reward += step_reward
            
            # Episode done
            if (i + 1) % episode_len == 0:
                Recoder.stop_recording()
                audio_data = Recoder.get_current_buffer()
                audio_data = audio_data[:882 * episode_len].squeeze()
                ref_audio = ref_audio[:882 * episode_len]
                Recoder.clear_buffer()

                if not os.path.exists("result/record_audios"):
                    os.makedirs("result/record_audios")
                wavio.write(f"result/record_audios/episode_{i}.wav", audio_data, rate=samplerate, sampwidth=4)
                # audio_data, rate = librosa.load(f"result/record_audios/episode_{i}.wav", sr=44100)

                max_amp = np.max(abs(audio_data))
                hit_amp_th = 0.0155

                # if max_amp > hit_amp_th: # calculate episode rewards only when the sound is load enough
                audio_data[abs(audio_data) < hit_amp_th] = 1e-5
                onset_reward, timing_reward, hit_reward = onset_rewards(audio_data, ref_audio, ref_sr)

                done_rewards = self.w_onset_rew * onset_reward \
                                + self.w_timing_rew * timing_reward \
                                + self.w_hit_rew * hit_reward
                
                step_reward += done_rewards
                episode_reward += done_rewards

                episode_rewards.append(episode_reward)
                epi_timing_rewards.append(timing_reward * self.w_timing_rew)
                epi_onset_rewards.append(onset_reward * self.w_onset_rew)
                epi_hit_rewards.append(hit_reward * self.w_hit_rew)
                    
                # else:
                #     print(f"** Pysonic didn't hit the key! **")
                #     episode_rewards.append(episode_reward)
                #     epi_timing_rewards.append(0)
                #     epi_onset_rewards.append(0)
                #     epi_hit_rewards.append(0)
            
            obs_trajectory.append(obs)
            act_trajectory.append(curr_action)
            reward_trajectory.append(step_reward)
            buffer.put(obs, action, step_reward, val, log_prob)
            obs = next_obs

        info.update({"episode_rewards": episode_rewards, "epi_timing_rewards": epi_timing_rewards, "epi_onset_rewards": epi_onset_rewards, "epi_hit_rewards": epi_hit_rewards})

        return info


    def update(self):
        
        episode_len = self.record_duration * self.ros_rate # (50HZ)
        max_steps_per_sampling = self.n_epi * episode_len 
        buffer_size = max_steps_per_sampling # in PPO (on-policy), buffer size is same as the total steps per sampling

        PPO = PPOClass(obs_dim=self.obs_dim,
                        act_dim=self.act_dim,
                        h_dims=self.h_dims,
                        gamma=self.gamma,
                        lmbda=self.lmbda,
                        lr_actorcritic=self.lr_actorcritic,
                        clip_ratio=self.clip_ratio,
                        value_coef=self.value_coef,
                        entropy_coef=self.entropy_coef,
                        max_grad=self.max_grad,
                        beta_dist=self.beta_dist)

        PPOBuffer = PPOBufferClass(obs_dim=self.obs_dim,
                                    act_dim=self.act_dim,
                                    buffer_size=buffer_size)
        
        # Set random seed
        self.set_random_seed()

        # Real-world rollouts
        YorN = str(input("Do you want to roll out real-world? (y/n): "))
        if YorN.lower() != 'n':

            # Load reference audio
            # TODO: adjust amp_rate
            amp_rate = 20 # 20 is generalization value. Compare the amplitude of the reference sound and the generated sound and adjust the value.
            ref_audio, ref_sr = librosa.load(self.ref_audio_path) # load reference audio
            ref_audio = ref_audio / amp_rate 

            # initial set
            control_joint_pos = self.initial_pose # 6-fingertip joint angles

            print("Initial position: ", control_joint_pos)
            self.QPosPublisher.publish_once(control_joint_pos)
            print("Initial position published")

            for i in range(self.max_iter):
                # Set initial state for logging
                actor_loss_ls = []
                critic_loss_ls = []
                total_loss_ls = []

                # Sample n trajectories, total steps = n * episode_len
                if (i + 1) % 5 == 0:
                    self.max_vel = min(self.max_vel * self.velocity_free_coef, 5)
                    self.min_vel = max(self.min_vel * self.velocity_free_coef, -5)
                rewards_info = self.sample_trajectory(PPO, PPOBuffer, max_steps_per_sampling, episode_len,
                                                                          min_vel=self.min_vel, max_vel=self.max_vel, samplerate=self.samplerate)
                info = self.logger.episode_reward_stat(rewards_info)

                # PPO training update
                for _ in range(self.k_epoch):
                    mini_batch_data = PPOBuffer.get_mini_batch(mini_batch_size=self.mini_batch_size) # split data into different subsets
                    n_mini_batch = len(mini_batch_data)
                    for k in range(n_mini_batch):
                        advantage_batch = mini_batch_data[k]['advantage']
                        advantage_batch = (advantage_batch - np.squeeze(np.mean(advantage_batch, axis=0))) / (np.squeeze(np.std(advantage_batch, axis=0)) + 1e-8)
                        
                        advantage_batch = np2torch(advantage_batch)
                        obs_batch = np2torch(mini_batch_data[k]['obs'])
                        act_batch = np2torch(mini_batch_data[k]['action'])
                        log_prob_batch = np2torch(mini_batch_data[k]['log_prob'])
                        return_batch = np2torch(mini_batch_data[k]['return'])

                        actor_loss, critic_loss, total_loss = PPO.update(obs_batch, act_batch, log_prob_batch, advantage_batch, return_batch)
                        actor_loss_ls.append(actor_loss.numpy())
                        critic_loss_ls.append(critic_loss.numpy())
                        total_loss_ls.append(total_loss.numpy())
                PPOBuffer.clear()      
                info.update({"actor_loss": np.mean(actor_loss_ls), "critic_loss": np.mean(critic_loss_ls), "total_loss": np.mean(total_loss_ls)})
                # Log trajectory rewards, actor loss, critic loss, total loss
                self.logger.log(info)


if __name__ == "__main__":
    sr = SoundRecorder()
    sr.start_recording()
    time.sleep(0.02)
    start_time = time.time()
    for i in range(100):
        time.sleep(0.02)
        cur_time = time.time()
        record_list = sr.get_current_buffer()
        print("time_elapse:", cur_time - start_time)
        print("length of list", len(record_list))
        print("\n")
