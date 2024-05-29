import torch
from rl.ppo import *
from ros_func.publisher import QPosPublisher
from utils.sound_recorder import SoundRecorder
from utils.psyonic_func import get_velocity, vel_clip_action, get_acceleration, \
    beta_dist_to_action_space, action_space_to_beta_dist, norm_to_action_space, normilize_obs
import argparse, os, sys, wandb, copy, rospy, time
from utils.eval_result import visualize_audio
from utils.reward_functions import dtw_reward, onset_hit_reward, assign_rewards_to_episode, move_penalty
import librosa

parser = argparse.ArgumentParser()
parser.add_argument('--obs_dim', type=int, default=2)
parser.add_argument('--act_dim', type=int, default=1)
parser.add_argument('--beta_dist', action="store_true")
parser.add_argument('--step', type=int, default=100)
parser.add_argument('--model_path', type=str) # 910 almost worked
parser.add_argument('--ref_audio_path', type=str, default="ref_audio/xylophone/ref_hit1_clip.wav")

args = parser.parse_args()


def tf_pos_emb(time_step):
    return np.array([np.sin(time_step), np.cos(time_step)])

def exec_model():
    global args, prev_action, curr_action, initial_pose, pose_lower, pose_upper, qpospublisher
    PPO = PPOClass(obs_dim=args.obs_dim,
                            act_dim=args.act_dim,
                            h_dims=[64,64],
                            gamma=0.99,
                            lmbda=0.95,
                            lr_actorcritic=1e-5,
                            clip_ratio=0.2,
                            value_coef=0.5,
                            entropy_coef=0.01,
                            max_grad=0.5,
                            beta_dist=args.beta_dist)
    PPO.load_state_dict(torch.load(args.model_path))
    action_list = []

    for i in range(args.step):
        print(f"step {i}")
        obs = np.concatenate((prev_action, curr_action), axis=0)
        obs = normilize_obs(obs, total_timestep=args.step, min_pos=pose_lower[-1], max_pos=pose_upper[-1])
        obs = obs + tf_pos_emb(i % args.step)
        ori_action = PPO.get_best_action(obs)

        clipped_action = np.clip(ori_action, -1, 1)
        action = norm_to_action_space(clipped_action)

        prev_action = curr_action
        if args.beta_dist:
            curr_action = beta_dist_to_action_space(action, pose_lower, pose_upper)
        else:
            curr_action = np.clip(action, pose_lower, pose_upper)

        # print(curr_action)
        action_list.append(curr_action)
        publish_pose = np.concatenate((initial_pose[:-1], curr_action), axis=0)
        qpospublisher.publish_once(publish_pose)

    return np.array(action_list).squeeze()


def exec_policy():
    global initial_pose, qpospublisher
    desired_actions = np.zeros((100,))
    desired_actions.fill(-10)
    # ===================desired movement=========================
    desired_actions[0:3] = -50

    # ===================shaking movement=========================
    # desired_actions[0::2] = -50
    # desired_actions[1::2] = -10

    # ===================only motor noise=========================
    # desired_actions[0::2] = -30
    # desired_actions[1::2] = -20

    # ===================hit at the end===========================
    # desired_actions[99] = -50
    # print(desired_actions)

    for i in range(100):
        pose_cmd = np.concatenate((initial_pose[:-1], [desired_actions[i]]), axis=0)
        print(pose_cmd)
        qpospublisher.publish_once(pose_cmd)

    return desired_actions

Recorder = SoundRecorder(samplerate=44100, audio_device=None)
initial_pose = np.array([50, 70, 110, 115, 50, -10])
pose_upper = np.array([-10])
pose_lower = np.array([-50])
rospy.init_node('psyonic_for_real', anonymous=True)
# QPosPublisher.publish_once(initial_pose)
qpospublisher = QPosPublisher()
qpospublisher.publish_once(initial_pose)
prev_action = initial_pose[-1:] # pre position
curr_action = initial_pose[-1:] # current position
Recorder.start_recording()

if args.model_path:
    action_list = exec_model()
else:
    action_list = exec_policy()

print(action_list)
time.sleep(0.1)
Recorder.stop_recording()

ref_audio, ref_sr = librosa.load(args.ref_audio_path, sr=44100) # load reference audio
rec_audio = Recorder.get_current_buffer()

rec_audio[abs(rec_audio) < 0.04] = 0
ref_audio[abs(ref_audio) < 0.04] = 0
amp_reward_list, hit_reward_list, timing_reward_list = assign_rewards_to_episode(ref_audio, 
                                                                                 rec_audio.squeeze(), 
                                                                                 100)

dtw_rwd = dtw_reward(ref_audio, rec_audio.squeeze())

move_penalty_list = move_penalty(action_list)
# D = np.abs(librosa.stft(ref_audio))
# times = librosa.times_like(D, sr=44100)
# print(D.shape)
# print(times)
print("hit_rew ", np.sum(hit_reward_list))
print("timing_rew ", np.sum(timing_reward_list))
print("move_rew ", np.sum(move_penalty_list))
print("dtw_rew: ", dtw_rwd)

visualize_audio(ref_audio, rec_audio, ref_sr, block=True)