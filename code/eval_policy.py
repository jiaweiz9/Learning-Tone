import torch
from rl.ppo import *
from ros_func.publisher import QPosPublisher
from utils.sound_recorder import SoundRecorder
from utils.psyonic_func import get_velocity, vel_clip_action, get_acceleration, \
    beta_dist_to_action_space, action_space_to_beta_dist, norm_to_action_space, normilize_obs
import argparse, os, sys, wandb, copy, rospy, time
from utils.eval_result import visualize_audio
import librosa

parser = argparse.ArgumentParser()
parser.add_argument('--obs_dim', type=int, default=3)
parser.add_argument('--act_dim', type=int, default=1)
parser.add_argument('--beta_dist', action="store_true")
parser.add_argument('--step', type=int, default=100)
parser.add_argument('--model_path', type=str, default="result/ppo/weights/PPO_960.pth") # 910 almost worked
parser.add_argument('--ref_audio_path', type=str, default="ref_audio/xylophone/ref_hit1_clip.wav")

args = parser.parse_args()

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

Recoder = SoundRecorder(samplerate=44100, audio_device=None)
initial_pose = np.array([50, 70, 110, 115, 50, -10])
pose_upper = np.array([-10])
pose_lower = np.array([-50])
rospy.init_node('psyonic_for_real', anonymous=True)
# QPosPublisher.publish_once(initial_pose)
qpospublisher = QPosPublisher()
qpospublisher.publish_once(initial_pose)
prev_action = initial_pose[-1:] # pre position
curr_action = initial_pose[-1:] # current position
Recoder.start_recording()

for i in range(args.step):
    print(f"step {i}")
    obs = np.concatenate((np.array([i]), prev_action, curr_action), axis=0)
    obs = normilize_obs(obs, total_timestep=args.step, min_pos=pose_lower[-1], max_pos=pose_upper[-1])
    ori_action = PPO.get_best_action(obs)

    clipped_action = np.clip(ori_action, -1, 1)
    action = norm_to_action_space(clipped_action)

    prev_action = curr_action
    if args.beta_dist:
        curr_action = beta_dist_to_action_space(action, pose_lower, pose_upper)
    else:
        curr_action = np.clip(action, pose_lower, pose_upper)

    print(curr_action)
    publish_pose = np.concatenate((initial_pose[:-1], curr_action), axis=0)
    qpospublisher.publish_once(publish_pose)


ref_audio, ref_sr = librosa.load(args.ref_audio_path, sr=44100) # load reference audio
Recoder.stop_recording()
rec_audio = Recoder.get_current_buffer()

visualize_audio(ref_audio, rec_audio, ref_sr)