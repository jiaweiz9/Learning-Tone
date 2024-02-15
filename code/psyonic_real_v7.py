import wandb, os, copy, sys, rospy, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.psyonic_func import *
from rl.ppo import *
from ros_func.publisher import QPosPublisher
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import wavio
import threading
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class SoundRecorder():
    def __init__(self, samplerate=44100, audio_device=None):
        self.stream = None
        self.is_recording = False
        self.recording_list = []  # Using list instead of np.array for efficiency during recording
        self.sample_rate = samplerate
        self.recording_thread = None
        self.lock = threading.Lock()
        self.audio_idx = audio_device

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
        else:
            print("Already recording!")

    def _record_audio(self):
        chunk_size = 882  # 20ms
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', device=self.audio_idx) as stream:
            print('Starting recording...')
            while self.is_recording:
                audio_chunk, overflowed = stream.read(chunk_size)
                if overflowed:
                    print("Audio buffer overflowed! Some audio might be lost.")
                with self.lock:
                    self.recording_list.append(audio_chunk)
            print("Recording stopped!")

    def stop_recording(self):
        # Stops the ongoing audio recording.
        self.is_recording = False
        if self.recording_thread is not None:
            self.recording_thread.join()
            self.recording_thread = None

    def get_current_buffer(self):
        with self.lock:
            if self.recording_list:
                # Convert list of chunks to a single numpy array
                return np.concatenate(self.recording_list, axis=0)
            else:
                print("Recording hasn't started yet!")
                return None

class PsyonicForReal():
    def __init__(self,
                 name = "Learning Make Well-Sound for Real Robot",
                 ref_audio_path = 'ref_audio/ref_audio_human_1.wav',
                 out_min = 0.087,
                 out_max = 1.047,
                 seed = 111,
                 ros_rate=50,
                 device_idx=0):
        self.ros_rate = ros_rate
        self.ref_audio_path = ref_audio_path
        self.seed = seed
        self.name = name
        self.out_min = out_min
        self.out_max = out_max
        self.device = torch.device("cuda:{}".format(device_idx) if torch.cuda.is_available() else "cpu")
        self.set_ros()

    # RL Functions
    def set_random_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def set_ros(self):
        rospy.init_node('psyonic_for_real', anonymous=True)
        self.QPosPublisher = QPosPublisher()
        
    # Calculation Functions
    def get_velocity(self, prev_action, curr_action):
        vel = (curr_action - prev_action)/(1/self.ros_rate) # 50HZ, ros_rate
        return vel
    def vel_clip_action(self,prev_action,action,min_vel=-8.0,max_vel=8.0):
        vel = self.get_velocity(prev_action, action)
        vel_clip = np.clip(vel,min_vel,max_vel)
        delta_action = vel_clip*(1/(self.ros_rate))
        curr_action = prev_action + delta_action
        return curr_action, vel_clip
    def get_acceleration(self, prev_vel, curr_vel):
        acc = (curr_vel - prev_vel)/(1/self.ros_rate)
        return acc

    def update(self, 
               max_iter = 500,
               ros_rate=50,
               record_duration=1,
               n_epi = 12,
               k_epoch = 10,
               max_pos = 1.047, # max joint position radian
               obs_dim = 6, # one-finger joint position, velocity, acceleration, mean_amp
               act_dim = 1, # one-finger joint position
               h_dims = [128, 128],
               gamma = 0.99,
               lmbda = 0.95,
               lr_actorcritic = 3e-4,
               clip_ratio = 0.2,
               value_coef = 0.5,
               entropy_coef = 0.01,
               max_grad = 0.5,
               samplerate = 44100,
               min_vel = -8.0,
               max_vel =8.0,
               SAVE_WEIGHTS = True,
               weight_path = None,
               weight_iter_num = None,
               WANDB = False,
               folder = None,
               args = None):
        
        # Set RL agent
        max_step = record_duration * ros_rate # (50HZ)
        buffer_size = max_step * n_epi
        mini_batch_size = int(buffer_size / max_step)
        n_step_per_update=buffer_size

        PPO = PPOClass(max_pos=max_pos,
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        h_dims=h_dims,
                        gamma=gamma,
                        lmbda=lmbda,
                        lr_actorcritic=lr_actorcritic,
                        clip_ratio=clip_ratio,
                        value_coef=value_coef,
                        entropy_coef=entropy_coef,
                        max_grad=max_grad)

        if weight_path is not None and weight_iter_num is not None:
            actor_weight_path = os.path.join(weight_path, f'psyonic_actor_{weight_iter_num}.pth')
            critic_weight_path = os.path.join(weight_path, f'psyonic_critic_{weight_iter_num}.pth')
            log_std_path = os.path.join(weight_path, f'psyonic_log_std_{weight_iter_num}.pth')

            PPO.actor.load_state_dict(torch.load(actor_weight_path))
            PPO.critic.load_state_dict(torch.load(critic_weight_path))
            PPO.log_std.data = torch.load(log_std_path)
            print(f"Load weight from {weight_path} and iter_num {weight_iter_num} successfully!")
        else:
            weight_iter_num = 0
        PPOBuffer = PPOBufferClass(obs_dim=obs_dim,
                                    act_dim=act_dim,
                                    buffer_size=buffer_size)
        # Set initial state
        one_epi_reward = 0
        epi_reward = 0
        epi_cnt = 0
        iter_cnt = 0
        actor_loss_ls = []
        critic_loss_ls = []
        total_loss_ls = []

        # Set random seed
        self.set_random_seed()

        # Set WandB
        if WANDB:
            wandb.init(project='making-tapping-sound', entity='taemoon-jeong', name=str(folder)+'-'+str(self.seed))
            wandb.config.update(args)

        # Real-world rollouts
        YorN = str(input("Do you want to roll out real-world? (y/n): "))
        if YorN.lower() != 'n':
            # Reward weight, we have different reward scale for each reward, so we need to tune the value
            w_amp_rew = 1
            w_onset_rew = 1e2
            w_timing_rew = 1e2 
            w_hit_rew = 1e2

            # Load reference audio
            amp_rate = 20 # 20 is generalization value. Compare the amplitude of the reference sound and the generated sound and adjust the value.
            ref_audio, ref_sr = librosa.load(self.ref_audio_path) # load reference audio
            ref_audio = ref_audio / amp_rate 

            # Set initial observation
            control_joint_pos = np.zeros(6) + self.out_min # 6-fingertip joint angles
            self.QPosPublisher.publish_once(control_joint_pos)
            obs = np.zeros(obs_dim)

            # Set initial trajectory
            pos_traj = [self.out_min]
            vel_traj = [0]
            acc_traj = [0]
            n_step = 0
            for t in range(1, buffer_size*max_iter + 1):
                n_step += 1
                ref_audio_info = ref_audio[441*(n_step-1):441*(n_step)]
                action, log_prob, val = PPO.get_action(obs)
                prev_action = pos_traj[n_step-1]
                curr_action, vel_clip = self.vel_clip_action(prev_action, action, min_vel=min_vel, max_vel=max_vel) # radian
                curr_action = np.clip(curr_action, self.out_min, self.out_max)
                control_joint_pos[0] = curr_action * (180./3.14)
                
                # Start recording
                if n_step == 1:
                    acc_traj.append(0)
                    Recoder = SoundRecorder(samplerate=samplerate, audio_device=None) # Bug fixed!! audio_devce=None is to use default connected device
                    Recoder.start_recording()
                    time.sleep(0.02)
                else:
                    prev_vel = vel_traj[n_step-2]
                    curr_vel = vel_clip
                    acc_traj.append(self.get_acceleration(prev_vel, curr_vel)[0])
                pos_traj.append(curr_action[0])
                vel_traj.append(vel_clip[0])
                self.QPosPublisher.publish_once(control_joint_pos) # Publish action 0.02 sec
                audio_data = Recoder.get_current_buffer() # get current sound data
                audio_data_step = audio_data[882*(n_step-1):882*(n_step)] # time window slicing 0.02sec
                
                # Calculate mean amplitude
                amp_scale = 1e2 # tune this value each audio because of the small amplitude value
                mean_amp = np.mean(abs(audio_data_step))*amp_scale
                if np.isnan(mean_amp):
                    mean_amp = 0
                mean_ref_amp = np.mean(abs(ref_audio_info))*amp_scale
                gap_amp = np.abs(mean_amp-mean_ref_amp)

                # Eq(1). Amplitude reward
                amp_reward = np.exp(-gap_amp)
                print(f"**Amplitude Reward:{w_amp_rew*amp_reward}")

                # Set next observation
                next_obs = np.array([pos_traj[n_step-1],pos_traj[n_step],vel_traj[n_step-1],vel_traj[n_step],acc_traj[n_step], mean_amp])
                
                # Episode done
                if n_step == max_step:
                    epi_cnt += 1
                    Recoder.stop_recording()
                    audio_data = Recoder.get_current_buffer()
                    wavio.write(f"result/record_audios/episode_{epi_cnt}.wav", audio_data, rate=samplerate, sampwidth=3)
                    audio_data, rate = librosa.load(f"result/record_audios/episode_{epi_cnt}.wav")
                    max_amp = np.max(abs(audio_data))
                    hit_amp_th = 0.0155 # hit amplitude threshold, Set appropriate value
                    if max_amp > hit_amp_th:
                        audio_data[abs(audio_data)<hit_amp_th] = 1e-5
                        audio_onset = audio_data / np.max(audio_data)
                        audio_ref_onset = ref_audio / np.max(ref_audio)

                        # Reference sound Onset
                        onset_env_ref = librosa.onset.onset_strength(y=audio_ref_onset, sr=ref_sr)
                        norm_onset_env_ref = onset_env_ref / np.max(onset_env_ref)
                        onset_frames_ref = librosa.onset.onset_detect(y=audio_ref_onset, sr=ref_sr)
                        beat_cnt_ref = onset_frames_ref.size
                        onset_times_ref = librosa.frames_to_time(onset_frames_ref,sr=ref_sr)

                        # Generated sound Onset
                        onset_env = librosa.onset.onset_strength(y=audio_onset, sr=ref_sr)
                        norm_onset_env = onset_env / np.max(onset_env)
                        onset_frames = librosa.onset.onset_detect(y=audio_onset, sr=ref_sr)
                        beat_cnt = onset_frames.size
                        onset_times = librosa.frames_to_time(onset_frames,sr=ref_sr)

                        # Eq(2). Onset strength reward
                        dtw_onset, _ = fastdtw(norm_onset_env, norm_onset_env_ref) # Onset DTW
                        onset_reward = (-dtw_onset)
                        # Eq(3). Onset timing reward
                        timing_reward = np.exp(-euclidean(onset_times_ref,onset_times))
                        # Eq(4). Hit reward
                        if beat_cnt_ref == beat_cnt:
                            hit_reward = beat_cnt
                        else:
                            hit_reward = 0
                            
                        print(f"**Onset Strength Reward:{w_onset_rew*onset_reward}")
                        print(f"**Onset Timing Reward:{w_timing_rew*timing_reward}")
                        print(f"**Hit Reward:{w_hit_rew*hit_reward}")
                    else:
                        onset_reward = 0
                        timing_reward = 0
                        hit_reward = 0
                        print(f"** You didn't touch to drum pad! **")

                    # Total Reward
                    total_reward = w_amp_rew*amp_reward + w_onset_rew*onset_reward + w_timing_rew*timing_reward + w_hit_rew*hit_reward
                    epi_reward += total_reward
                    one_epi_reward += total_reward
                    print(f"{epi_cnt} Episode_reward: {one_epi_reward}")
                    PPOBuffer.put(obs, action, total_reward, val, log_prob)
                    obs = next_obs
                    last_val = PPO.get_val(obs)
                    PPOBuffer.get_gae_batch(gamma=PPO.gamma,
                                            lmbda=PPO.lmbda,
                                            last_val=last_val)
                    # Set initial observation
                    control_joint_pos = np.zeros(6) + self.out_min
                    self.QPosPublisher.publish_once(control_joint_pos)
                    pos_traj = [self.out_min]
                    vel_traj = [0]
                    acc_traj = [0]
                    self.audio_data = None
                    obs = np.zeros(obs_dim)
                    n_step = 0
                    one_epi_reward = 0
                    time.sleep(0.2)
                else:
                    total_reward = w_amp_rew*amp_reward
                    epi_reward += total_reward
                    one_epi_reward += total_reward
                    PPOBuffer.put(obs, action, total_reward, val, log_prob)
                    obs = next_obs
                    prev_action = action
                if t % n_step_per_update == 0:
                    iter_cnt += 1
                    for _ in range(k_epoch):
                        mini_batch_data = PPOBuffer.get_mini_batch(mini_batch_size=mini_batch_size)
                        n_mini_batch = len(mini_batch_data)
                        for k in range(n_mini_batch):
                            obs_batch = mini_batch_data[k]['obs']
                            act_batch = mini_batch_data[k]['action']
                            log_prob_batch = mini_batch_data[k]['log_prob']
                            advantage_batch = mini_batch_data[k]['advantage']
                            advantage_batch = (advantage_batch - np.squeeze(np.mean(advantage_batch, axis=0))) / (np.squeeze(np.std(advantage_batch, axis=0)) + 1e-8)
                            return_batch = mini_batch_data[k]['return']

                            obs_batch = np2torch(obs_batch)
                            act_batch = np2torch(act_batch)
                            log_prob_batch = np2torch(log_prob_batch)
                            advantage_batch = np2torch(advantage_batch)
                            return_batch = np2torch(return_batch)

                            actor_loss, critic_loss, total_loss = PPO.update(obs_batch, act_batch, log_prob_batch, advantage_batch, return_batch)
                            actor_loss_ls.append(actor_loss.numpy())
                            critic_loss_ls.append(critic_loss.numpy())
                            total_loss_ls.append(total_loss.numpy())
                    PPOBuffer.clear()      
                    mean_ep_reward = epi_reward / epi_cnt
                    epi_reward, epi_cnt = 0, 0
                    # Wandb
                    if WANDB:
                        wandb.log({"Iter": iter_cnt, "AVG_REWARD": mean_ep_reward, "ACTOR_LOSS": np.mean(actor_loss_ls), "CRITIC_LOSS": np.mean(critic_loss_ls), "TOTAL_LOSS": np.mean(total_loss_ls)})
                    print(f"Iter={iter_cnt}, AVG_REWARD={mean_ep_reward:.2f}, ACTOR_LOSS={np.mean(actor_loss_ls):.2f}, CRITIC_LOSS={np.mean(critic_loss_ls):.2f}, TOTAL_LOSS={np.mean(total_loss_ls):.2f}")
                    actor_loss_ls = []; critic_loss_ls = []; total_loss_ls = []
                    if SAVE_WEIGHTS:
                        if iter_cnt % 5 == 0:
                            if not os.path.exists("result/weights"):
                                os.makedirs("result/weights")
                            torch.save(PPO.actor.state_dict(), f"result/ppo/weights/psyonic_actor_{iter_cnt+weight_iter_num}.pth")
                            torch.save(PPO.critic.state_dict(), f"result/ppo/weights/psyonic_critic_{iter_cnt+weight_iter_num}.pth")
                            torch.save(PPO.log_std, f"result/ppo/weights/psyonic_log_std_{iter_cnt+weight_iter_num}.pth")

            
