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
from utils.eval_result import save_vis_reward_components


class PsyonicForReal():
    def __init__(self, args):

        # reward setting
        self.w_amp_rew = 0
        self.w_dtw_rew = 100
        self.w_timing_rew = 100
        self.w_hit_rew = 2

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
        self.pose_upper = np.array([-0])
        self.pose_lower = np.array([-30])
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
        log_config = {"w_amp_rew": self.w_amp_rew, "w_hit_rew": self.w_hit_rew,
                      "n_epi": self.n_epi, "mini_batch_size": self.mini_batch_size,}

        self.logger = Logger(args.WANDB, log_config, resume = self.reload_iter > 0)
        
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

        ref_audio, ref_sr = librosa.load(self.ref_audio_path, sr=44100) # load reference audio
        Recoder = SoundRecorder(samplerate=samplerate, audio_device=None) # Bug fixed!! audio_devce=None is to use default connected device
        rollouts_rew_total = 0
        rollouts_rew_amp = 0
        rollouts_rew_hit = 0

        for i in range(max_step):

            if i % episode_len == 0:
                prev_action = self.initial_pose[-1:] # pre position
                curr_action = self.initial_pose[-1:] # current position
                obs = np.concatenate((np.array([i]), prev_action, curr_action), axis=0)
                # Start recording
                Recoder.start_recording()

            else:
                prev_action = curr_action
                obs = next_obs
            
            # Get action from actor
            action, log_prob, val = PPO_agent.get_action(obs)
            
            if self.beta_dist:
                curr_action = beta_dist_to_action_space(action, self.pose_lower, self.pose_upper)
            else:
                curr_action = np.clip(action, self.pose_lower, self.pose_upper)

            # Clip action based on velocity
            curr_action, vel_clip = vel_clip_action(prev_action, curr_action, min_vel=min_vel, max_vel=max_vel, ros_rate=self.ros_rate) # radian
            
            # print("action_after_clip: ", curr_action)

            publish_pose = np.concatenate((self.initial_pose[:-1], curr_action), axis=0)
            self.QPosPublisher.publish_once(publish_pose) # Publish action 0.02 sec

            # Get audio data

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
                # hit_amp_th = 0.005
                audio_data[abs(audio_data) < hit_amp_th] = 1e-5
                # ref_audio[abs(ref_audio) < hit_amp_th] = 1e-5

                Recoder.clear_buffer()
                print("audio_data shape: ", audio_data.shape)
                print("ref_audio shape: ", ref_audio.shape)

                if not os.path.exists("result/record_audios"):
                    os.makedirs("result/record_audios")
                episode_num = (i + 1) // episode_len + iter * self.n_epi
                if episode_num % 50 == 0:
                    wavio.write(f"result/record_audios/episode_{episode_num}.wav", audio_data, rate=samplerate, sampwidth=4)

                max_amp = np.max(abs(audio_data))

                # if max_amp > hit_amp_th: # calculate episode rewards only when the sound is loud enough
                amp_reward_list, hit_reward_list = assign_rewards_to_episode(ref_audio, audio_data, episode_len)
                
                reward_trajectory = amp_reward_list * self.w_amp_rew \
                                    + hit_reward_list * self.w_hit_rew 

                assert len(reward_trajectory) == episode_len, len(reward_trajectory)
                assert len(obs_trajectory) == episode_len, len(obs_trajectory)
                assert len(act_trajectory) == episode_len, len(act_trajectory)

                if episode_num % 10 == 0:
                    save_vis_reward_components(ref_audio, audio_data, episode_len, sr=44100, 
                                        rewards_dict={
                                            "Amplitude Reward": amp_reward_list * self.w_amp_rew,
                                            "Hit Reward": hit_reward_list * self.w_hit_rew,
                                            }, 
                                            img_path=f"result/vis_rewards/episode_{episode_num}.png")

                info["rewards"] = np.sum(reward_trajectory)
                # epi_timing_rewards.append()
                info["hit_rewards"] = (np.sum(hit_reward_list) * self.w_hit_rew)
                info["amp_rewards"] = (np.sum(amp_reward_list) * self.w_amp_rew)

                self.logger.log(info, True)
                rollouts_rew_total += np.sum(reward_trajectory)
                rollouts_rew_amp += np.sum(amp_reward_list) * self.w_amp_rew
                rollouts_rew_hit += np.sum(hit_reward_list) * self.w_hit_rew


                for obs, action, step_reward, val, log_prob in zip(obs_trajectory, act_trajectory, reward_trajectory, val_trajectory, log_prob_trajectory):
                    buffer.put(obs, action, step_reward, val, log_prob)
                
                last_val = PPO_agent.get_val(next_obs)

                buffer.get_gae_batch(gamma = self.gamma, lmbda = self.lmbda, last_val = last_val)

                obs_trajectory = []
                act_trajectory = []
                reward_trajectory = []
                val_trajectory = []
                log_prob_trajectory = []

                self.QPosPublisher.publish_once(self.initial_pose)

        return rollouts_rew_total, rollouts_rew_amp, rollouts_rew_hit


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

            # initial set
            control_joint_pos = self.initial_pose # 6-fingertip joint angles

            print("Initial position: ", control_joint_pos)
            self.QPosPublisher.publish_once(control_joint_pos)
            print("Initial position published")

            if self.reload_iter > 0:
                PPO.load_state_dict(torch.load(f"result/ppo/weights/PPO_{self.reload_iter}.pth"))

            for i in range(self.reload_iter, self.max_iter):
                # Set initial state for logging
                actor_loss_ls = []
                critic_loss_ls = []
                total_loss_ls = []
                training_info = {}

                print("=================Iteration: ", i, "=================")
                # Sample n trajectories, total steps = n * episode_len
                if (i + 1) % 5 == 0:
                    self.max_vel = min(self.max_vel * self.velocity_free_coef, 5)
                    self.min_vel = max(self.min_vel * self.velocity_free_coef, -5)
                rollouts_rew_total, rollouts_rew_amp, rollouts_rew_hit = self.sample_trajectory(i, PPO, PPOBuffer, max_steps_per_sampling, episode_len, min_vel=self.min_vel, max_vel=self.max_vel, samplerate=self.samplerate)
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

                training_info= {"actor_loss": np.mean(actor_loss_ls), "critic_loss": np.mean(critic_loss_ls), "total_loss": np.mean(total_loss_ls)}
                # Log trajectory rewards, actor loss, critic loss, total loss
                self.logger.log(training_info, True)


                training_info.update({"rollouts_rew_total": rollouts_rew_total / self.n_epi , "rollouts_rew_amp": rollouts_rew_amp / self.n_epi, "rollouts_rew_hit": rollouts_rew_hit / self.n_epi})
                self.logger.log(training_info)

                if self.SAVE_WEIGHTS and (i + 1) % self.weight_iter_num == 0:
                    torch.save(PPO.state_dict(), f"result/ppo/weights/PPO_{i + 1}.pth")

