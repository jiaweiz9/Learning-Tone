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
from utils.reward_functions import amplitude_reward, onset_rewards, assign_rewards_to_episode
from utils.psyonic_func import get_velocity, vel_clip_action, get_acceleration, beta_dist_to_action_space, action_space_to_beta_dist
from utils.logger import Logger
from utils.saveload_model import save_model, load_model


class PsyonicForReal():
    def __init__(self, args):

        # reward setting
        self.w_amp_rew = 1
        self.w_dtw_rew = 1e-1
        self.w_timing_rew = 1e3
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
        self.reload_iter = args.reload_iter

        # action setting
        self.initial_pose = np.array([105, 105, 105, 110, 70, -0])
        # self.pose_upper = np.array([105, 105, 105, 110, 70, -10])
        # self.pose_lower = np.array([95, 95, 95, 100, 65, -35])
        self.pose_upper = np.array([-0])
        self.pose_lower = np.array([-35])
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
        log_config = {"w_amp_rew": self.w_amp_rew, "w_dtw_rew": self.w_dtw_rew, "w_hit_rew": self.w_hit_rew,
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
    
    
    def sample_trajectory(self, iter, PPO_agent, buffer, max_step, episode_len, min_vel, max_vel, samplerate):
        obs_trajectory = []
        act_trajectory = []
        reward_trajectory = []
        val_trajectory = []
        log_prob_trajectory = []
        info = {}

        episode_rewards = []
        epi_dtw_rewards = []
        epi_timing_rewards = []
        epi_amp_rewards = []
        epi_hit_times_rewards = []
        ref_audio, ref_sr = librosa.load(self.ref_audio_path, sr=44100) # load reference audio
        Recoder = SoundRecorder(samplerate=samplerate, audio_device=None) # Bug fixed!! audio_devce=None is to use default connected device

        for i in range(max_step):

            if i % episode_len == 0:
                prev_action = self.initial_pose[-1:] # pre position
                curr_action = self.initial_pose[-1:] # current position
                # shape (timestep, pre_position, curr_position, cur_amp) = (14,)
                obs = np.concatenate((np.array([i]), prev_action, curr_action), axis=0)

                # Start recording
                Recoder.start_recording()
                start_time = time.time()

            else:
                prev_action = curr_action
                obs = next_obs
            
            # Get action from actor
            action, log_prob, val = PPO_agent.get_action(obs)
            
            if self.beta_dist:
                curr_action = beta_dist_to_action_space(action, self.pose_lower, self.pose_upper)
            else:
                curr_action = np.clip(action, self.pose_lower, self.pose_upper)

            # print("action_before_clip: ", curr_action)
            # Clip action based on velocity
            curr_action, vel_clip = vel_clip_action(prev_action, curr_action, min_vel=min_vel, max_vel=max_vel, ros_rate=self.ros_rate) # radian
            
            # print("action_after_clip: ", curr_action)

            publish_pose = np.concatenate((self.initial_pose[:-1], curr_action), axis=0)
            self.QPosPublisher.publish_once(publish_pose) # Publish action 0.02 sec

            # Get audio data

            # ref_audio_info = ref_audio[882* (i % episode_len) : 882* ((i  % episode_len) + 1)]
            # while True:
            #     audio_data = Recoder.get_current_buffer() # get current sound data
            #     if len(audio_data) > 0:
            #         cur_time = time.time() if i == 0 else cur_time
            #         break
            # audio_data = Recoder.get_current_buffer()
            # audio_data_step = audio_data[882* (i % episode_len) : 882* ((i  % episode_len) + 1)] # time window slicing 0.02sec
            # audio_data_step = audio_data[-882 : ] # 0.02 sec
            # curr_amp = np.mean(np.abs(audio_data_step))

            # Set next observation
            # curr_amp = np.asarray(curr_amp).reshape(1)
            # print("curr_amp: ", curr_amp)
            next_obs = np.concatenate((np.array([i + 1]), prev_action, curr_action), axis=0)

            obs_trajectory.append(obs)
            act_trajectory.append(action_space_to_beta_dist(curr_action, self.pose_lower, self.pose_upper))
            val_trajectory.append(val)
            log_prob_trajectory.append(log_prob)
            
            
            # Episode done
            if (i + 1) % episode_len == 0:
                Recoder.stop_recording()
                audio_data = Recoder.get_current_buffer()

                # print("start waiting time", cur_time - start_time)
                audio_data = audio_data.squeeze()[8820:] # remove the waiting time
                ref_audio = ref_audio[:882 * episode_len]
                hit_amp_th = 0.0155
                audio_data[abs(audio_data) < hit_amp_th] = 1e-5

                Recoder.clear_buffer()
                print("audio_data shape: ", audio_data.shape)
                print("ref_audio shape: ", ref_audio.shape)

                if not os.path.exists("result/record_audios"):
                    os.makedirs("result/record_audios")
                episode_num = (i + 1) // episode_len + iter * self.n_epi
                wavio.write(f"result/record_audios/episode_{episode_num}.wav", audio_data, rate=samplerate, sampwidth=4)

                max_amp = np.max(abs(audio_data))

                # if max_amp > hit_amp_th: # calculate episode rewards only when the sound is load enough
                amp_reward_list, dtw_reward_list, hit_times_reward, hit_timing_reward = assign_rewards_to_episode(ref_audio, audio_data, episode_len)
                
                reward_trajectory = amp_reward_list * self.w_amp_rew \
                                    + dtw_reward_list * self.w_dtw_rew
                                    # + hit_reward_list * self.w_hit_rew
                
                reward_trajectory[-1] += self.w_timing_rew * hit_timing_reward + self.w_hit_rew * hit_times_reward

                print("Hit Timing Reward: ", hit_timing_reward)

                assert len(reward_trajectory) == episode_len, len(reward_trajectory)
                assert len(obs_trajectory) == episode_len, len(obs_trajectory)
                assert len(act_trajectory) == episode_len, len(act_trajectory)

                episode_rewards.append(np.sum(reward_trajectory))
                # epi_timing_rewards.append()
                epi_dtw_rewards.append(np.sum(dtw_reward_list) * self.w_dtw_rew)
                epi_timing_rewards.append(hit_timing_reward * self.w_timing_rew)
                epi_hit_times_rewards.append(hit_times_reward * self.w_hit_rew)
                epi_amp_rewards.append(np.sum(amp_reward_list) * self.w_amp_rew)

                for obs, action, step_reward, val, log_prob in zip(obs_trajectory, act_trajectory, reward_trajectory, val_trajectory, log_prob_trajectory):
                    buffer.put(obs, action, step_reward, val, log_prob)

                obs_trajectory = []
                act_trajectory = []
                reward_trajectory = []
                val_trajectory = []
                log_prob_trajectory = []

                # if (iter * self.n_epi + (i + 1) // episode_len) % 10 == 0:
                #     wavio.write(f"result/record_audios/episode_{iter * self.n_epi + (i + 1) // episode_len}.wav", 
                #                 audio_data, rate=samplerate, sampwidth=4)

            

        info.update({"episode_rewards": episode_rewards, 
                     "reward_components": {
                         "epi_amp_rewards": epi_amp_rewards, 
                         "epi_dtw_rewards": epi_dtw_rewards, 
                         "epi_timing_rewards": epi_timing_rewards,
                            "epi_hit_times_rewards": epi_hit_times_rewards
                     }
                     })

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
            # amp_rate = 20 # 20 is generalization value. Compare the amplitude of the reference sound and the generated sound and adjust the value.
            # ref_audio, ref_sr = librosa.load(self.ref_audio_path) # load reference audio
            # ref_audio = ref_audio / amp_rate 

            # initial set
            control_joint_pos = self.initial_pose # 6-fingertip joint angles

            print("Initial position: ", control_joint_pos)
            self.QPosPublisher.publish_once(control_joint_pos)
            print("Initial position published")

            if self.reload_iter > 0:
                PPO.load_state_dict(torch.load(f"result/weights/PPO_{self.reload_iter}.pth"))

            for i in range(self.max_iter):
                # Set initial state for logging
                actor_loss_ls = []
                critic_loss_ls = []
                total_loss_ls = []

                print("=================Iteration: ", i, "=================")
                # Sample n trajectories, total steps = n * episode_len
                if (i + 1) % 5 == 0:
                    self.max_vel = min(self.max_vel * self.velocity_free_coef, 5)
                    self.min_vel = max(self.min_vel * self.velocity_free_coef, -5)
                rewards_info = self.sample_trajectory(i, PPO, PPOBuffer, max_steps_per_sampling, episode_len,
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

                if self.SAVE_WEIGHTS and (i + 1) % self.weight_iter_num == 0:
                    torch.save(PPO.state_dict(), f"result/weights/PPO_{i + 1}.pth")


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
