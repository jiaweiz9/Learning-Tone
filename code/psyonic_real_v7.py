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
from utils.psyonic_func import get_velocity, vel_clip_action, get_acceleration


class PsyonicForReal():
    def __init__(self,
                 name = "Learning Make Well-Sound for Real Robot",
                 ref_audio_path = 'ref_audio/xylophone/ref_hit2.wav',
                 seed = 111,
                 ros_rate=50,
                 device_idx=0):
        self.ros_rate = ros_rate
        self.ref_audio_path = ref_audio_path
        self.seed = seed
        self.name = name
        # self.out_min = out_min
        # self.out_max = out_max
        self.initial_pose = np.array([105, 105, 105, 110, 70, -10]) * 3.14 / 180
        print("initial pose:", self.initial_pose)
        self.max_pose = np.array([95, 95, 95, 100, 65, -25]) * 3.14 / 180

        self.w_amp_rew = 1
        self.w_onset_rew = 1e2
        self.w_timing_rew = 1e2
        self.w_hit_rew = 1e2

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
        
    
    def sample_trajectory(self, PPO_agent, buffer, max_step, episode_len, max_pos, min_vel, max_vel, samplerate):
        obs_trajectory = []
        act_trajectory = []
        reward_trajectory = []

        episode_count = 0
        total_step = 0
        total_reward = 0

        for i in range(max_step):

            # For every new episode, we reset the position to the minimum value?, obersevation to zero
            if i % episode_len == 0:
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
            curr_action, vel_clip = vel_clip_action(prev_action, action, min_vel=min_vel, max_vel=max_vel, ros_rate=self.ros_rate) # radian

            curr_action = np.clip(curr_action, max_pos, self.initial_pose)

            control_joint_pos = curr_action * (180./3.14)
            self.QPosPublisher.publish_once(control_joint_pos) # Publish action 0.02 sec

            velocity = get_velocity(prev_action, curr_action, ros_rate=self.ros_rate)
            acceleration = get_acceleration(pre_velocity, velocity, ros_rate=self.ros_rate)
            pre_velocity = velocity

            # Get audio data
            cur_time = time.time()
            print("======step:", i)
            print("==time elapsed:", cur_time - start_time)

            ref_audio, ref_sr = librosa.load(self.ref_audio_path, sr=44100) # load reference audio
            print("=ref_audio_len: ", len(ref_audio))
            ref_audio_info = ref_audio[882* i : 882* (i + 1)]

            audio_data = Recoder.get_current_buffer() # get current sound data
            print("=record_audio_len:", len(audio_data))
            audio_data_step = audio_data[-882: ] # time window slicing 0.02sec
            
            print("=ref_audio_step:", len(ref_audio_info))
            print("=audio_data_step:", len(audio_data_step))

            amp_reward, mean_amp = amplitude_reward(audio_data_step, ref_audio_info, amp_scale=1e2)
            print(f"**Amplitude Reward:{self.w_amp_rew * amp_reward}")

            # Set next observation
            mean_amp = np.asarray(mean_amp).reshape(1)
            print("mean amp", mean_amp.shape)
            next_obs = np.concatenate((prev_action, curr_action, pre_velocity, velocity, acceleration, mean_amp), axis=0)

            onset_reward = 0
            timing_reward = 0
            hit_reward = 0
            
            # Episode done
            if (i + 1) % episode_len == 0:
                Recoder.stop_recording()
                audio_data = Recoder.get_current_buffer()
                Recoder.clear_buffer()

                if not os.path.exists("result/record_audios"):
                    os.makedirs("result/record_audios")
                wavio.write(f"result/record_audios/episode_{i}.wav", audio_data, rate=samplerate, sampwidth=4)
                audio_data, rate = librosa.load(f"result/record_audios/episode_{i}.wav", sr=44100)

                max_amp = np.max(abs(audio_data))
                hit_amp_th = 0.0155
                episode_count += 1

                if max_amp > hit_amp_th: # calculate episode rewards only when the sound is load enough
                    audio_data[abs(audio_data)<hit_amp_th] = 1e-5
                    onset_reward, timing_reward, hit_reward = onset_rewards(audio_data, ref_audio, ref_sr)

                    print(f"**Onset Strength Reward:{self.w_onset_rew * onset_reward}")
                    print(f"**Onset Timing Reward:{self.w_timing_rew * timing_reward}")
                    print(f"**Hit Reward:{self.w_hit_rew * hit_reward}")
                else:
                    
                    print(f"** You didn't touch to drum pad! **")
            
            # Total Reward
            step_reward = self.w_amp_rew * amp_reward \
                            + self.w_onset_rew * onset_reward \
                            + self.w_timing_rew * timing_reward \
                            + self.w_hit_rew * hit_reward
            
            obs_trajectory.append(obs)
            act_trajectory.append(curr_action)
            reward_trajectory.append(step_reward)
            buffer.put(obs, action, step_reward, val, log_prob)
            obs = next_obs
            total_step += 1
            total_reward += step_reward
        
        return episode_count, total_reward, total_step


    def update(self, 
                    max_iter = 100,
                    ros_rate=50,
                    record_duration=4,
                    n_epi = 10,     # total episode per sampling
                    k_epoch = 100,
                    mini_batch_size = 100,
                    obs_dim = 31, # 5-finger joint position dim(6)*2, velocity dim(6)*2, acceleration dim(6), mean_amp dim(1)
                    act_dim = 6, # 5-finger joint position
                    h_dims = [128, 128],
                    gamma = 0.99,
                    lmbda = 0.95,
                    lr_actorcritic = 3e-4,
                    clip_ratio = 0.2,
                    value_coef = 0.5,
                    entropy_coef = 0.01,
                    max_grad = 0.5,
                    samplerate = 44100,
                    min_vel = -1.0,
                    max_vel = 1.0,
                    SAVE_WEIGHTS = True,
                    weight_path = None,
                    weight_iter_num = None,
                    WANDB = False,
                    folder = None,
                    args = None):
        
        episode_len = record_duration * ros_rate # (50HZ)
        max_steps_per_sampling = n_epi * episode_len # default 2000
        buffer_size = max_steps_per_sampling # in PPO (on-policy), buffer size is same as the total steps per sampling

        PPO = PPOClass(obs_dim=obs_dim,
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
        
        # Set initial state for logging
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
            wandb.init(project='music', name=str(folder)+'-'+str(self.seed))
            wandb.config.update(args)

        # Real-world rollouts
        YorN = str(input("Do you want to roll out real-world? (y/n): "))
        if YorN.lower() != 'n':

            # Load reference audio
            amp_rate = 20 # 20 is generalization value. Compare the amplitude of the reference sound and the generated sound and adjust the value.
            ref_audio, ref_sr = librosa.load(self.ref_audio_path) # load reference audio
            ref_audio = ref_audio / amp_rate 

            # initial set
            control_joint_pos = self.initial_pose * (180./3.14) # 6-fingertip joint angles

            print("Initial position: ", control_joint_pos)
            self.QPosPublisher.publish_once(control_joint_pos)
            print("Initial position published")

            for i in range(max_iter):
                iter_cnt += 1
                # Sample n trajectories, total steps = n * episode_len
                epi_cnt, epi_reward, total_steps = self.sample_trajectory(PPO, PPOBuffer, max_steps_per_sampling, episode_len, self.max_pose, min_vel, max_vel, samplerate)
                assert total_steps == max_steps_per_sampling, "Total steps are expected to be {max_steps_per_sampling}, but it is {total_steps}"

                # PPO training update
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

                if WANDB:
                    wandb.log({"Iter": iter_cnt, "AVG_REWARD": mean_ep_reward, "ACTOR_LOSS": np.mean(actor_loss_ls), "CRITIC_LOSS": np.mean(critic_loss_ls), "TOTAL_LOSS": np.mean(total_loss_ls)})
                
                
                print(f"Iter={iter_cnt}, AVG_REWARD={mean_ep_reward:.2f}, ACTOR_LOSS={np.mean(actor_loss_ls):.2f}, CRITIC_LOSS={np.mean(critic_loss_ls):.2f}, TOTAL_LOSS={np.mean(total_loss_ls):.2f}")
                actor_loss_ls = []; critic_loss_ls = []; total_loss_ls = []
                if SAVE_WEIGHTS:
                    if iter_cnt % 5 == 0:
                        if not os.path.exists("result/ppo/weights"):
                            os.makedirs("result/ppo/weights")
                        torch.save(PPO.actor.state_dict(), f"result/ppo/weights/psyonic_actor_{iter_cnt+weight_iter_num}.pth")
                        torch.save(PPO.critic.state_dict(), f"result/ppo/weights/psyonic_critic_{iter_cnt+weight_iter_num}.pth")
                        torch.save(PPO.log_std, f"result/ppo/weights/psyonic_log_std_{iter_cnt+weight_iter_num}.pth")

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
