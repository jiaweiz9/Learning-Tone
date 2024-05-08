import torch
from rl.ppo import *
from ros_func.publisher import QPosPublisher
from utils.psyonic_func import get_velocity, vel_clip_action, get_acceleration, beta_dist_to_action_space, action_space_to_beta_dist
import argparse, os, sys

parser = argparse.ArgumentParser()
parser.add_argument('--obs_dim', type=int, default=3)
parser.add_argument('--act_dim', type=int, default=1)
parser.add_argument('--beta_dist', action="store_true")

args = parser.parse_args()

PPO = PPOClass(obs_dim=args.obs_dim,
                        act_dim=args.act_dim,
                        h_dims=args.h_dims,
                        gamma=args.gamma,
                        lmbda=args.lmbda,
                        lr_actorcritic=args.lr_actorcritic,
                        clip_ratio=args.clip_ratio,
                        value_coef=args.value_coef,
                        entropy_coef=args.entropy_coef,
                        max_grad=args.max_grad,
                        beta_dist=args.beta_dist)

PPO.load_state_dict()