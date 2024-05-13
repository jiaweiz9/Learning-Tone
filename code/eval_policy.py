import torch
from rl.ppo import *
from ros_func.publisher import QPosPublisher
from utils.sound_recorder import SoundRecorder
from utils.psyonic_func import get_velocity, vel_clip_action, get_acceleration, beta_dist_to_action_space, action_space_to_beta_dist
import argparse, os, sys, wandb, copy, rospy, time

parser = argparse.ArgumentParser()
parser.add_argument('--obs_dim', type=int, default=3)
parser.add_argument('--act_dim', type=int, default=1)
parser.add_argument('--beta_dist', action="store_true")
parser.add_argument('--step', type=int, default=200)
parser.add_argument('--model_path', type=str, default="result/ppo/weights/PPO_600.pth")

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
                        beta_dist=True)

PPO.load_state_dict(torch.load(args.model_path))

Recoder = SoundRecorder(samplerate=44100, audio_device=None)
initial_pose = np.array([105, 105, 105, 110, 70, -0])
pose_upper = np.array([-0])
pose_lower = np.array([-30])
rospy.init_node('psyonic_for_real', anonymous=True)
# QPosPublisher.publish_once(initial_pose)
qpospublisher = QPosPublisher()
qpospublisher.publish_once(initial_pose)
prev_action = initial_pose[-1:] # pre position
curr_action = initial_pose[-1:] # current position
# Recoder.start_recording()

for i in range(args.step):
    obs = np.concatenate((np.array([i]), prev_action, curr_action), axis=0)
    action = PPO.get_best_action(obs)

    prev_action = curr_action
    if args.beta_dist:
        curr_action = beta_dist_to_action_space(action, pose_lower, pose_upper)
    else:
        curr_action = np.clip(action, pose_lower, pose_upper)

    publish_pose = np.concatenate((initial_pose[:-1], curr_action), axis=0)
    print("step:", publish_pose)
    qpospublisher.publish_once(publish_pose)
