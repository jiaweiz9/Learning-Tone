import argparse, os, sys
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from psyonic_real_v7 import *

def main(args):
    Psyonic = PsyonicForReal(ref_audio_path=args.ref_audio_path,
                             ros_rate=args.ros_rate,
                             out_min=args.out_min,
                             out_max=args.out_max,
                             seed=args.seed)
    Psyonic.update(min_vel=args.min_vel,
                   max_vel=args.max_vel,
                   max_iter=args.max_iter,
                   ros_rate=args.ros_rate,
                   record_duration=args.record_duration,
                   n_epi=args.n_epi,
                   k_epoch=args.k_epoch,
                   max_pos=args.max_pos,
                   obs_dim=args.obs_dim,
                   act_dim=args.act_dim,
                   h_dims=args.h_dims,
                   gamma=args.gamma,
                   lmbda=args.lmbda,
                   lr_actorcritic=args.lr_actorcritic,
                   clip_ratio=args.clip_ratio,
                   value_coef=args.value_coef,
                   entropy_coef=args.entropy_coef,
                   max_grad=args.max_grad,
                   samplerate=args.samplerate,
                   WANDB=args.WANDB,
                   folder=args.folder,
                   weight_path=args.weight_path,
                   weight_iter_num=args.weight_iter_num,
                   SAVE_WEIGHTS=args.SAVE_WEIGHTS,
                   args=args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=111, type=int)
    parser.add_argument('--min_vel', type=float, default=-8.0)
    parser.add_argument('--max_vel', type=float, default=8.0)    
    parser.add_argument('--out_min', type=float, default=0.087) 
    parser.add_argument('--out_max', type=float, default=1.047)
    parser.add_argument('--ref_audio_path', type=str, default='ref_audio/xylophone/ref_hit2.wav')
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--ros_rate', type=int, default=50)
    parser.add_argument('--record_duration', type=int, default=1)
    
    parser.add_argument('--n_epi', type=int, default=1)
    parser.add_argument('--k_epoch', type=int, default=10)
    parser.add_argument('--max_pos', type=float, default=1.0)
    parser.add_argument('--obs_dim', type=int, default=31)
    parser.add_argument('--act_dim', type=int, default=6)
    
    parser.add_argument('--h_dims', nargs="+", type=int, default=[128, 128]) # change this term
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lmbda', type=float, default=0.95)

    parser.add_argument('--lr_actorcritic', type=float, default=3e-4) # change this term 0.003 ~ 5e-6
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--max_grad', type=float, default=0.5)
    parser.add_argument('--samplerate', type=int, default=44100) # Audio sample rate
    parser.add_argument('--WANDB', type=bool, default=True)
    parser.add_argument('--folder', type=str, default='psyonic-experiment')
    # parser.add_argument('--weight_path', type=str, default='result/ppo/weights/')
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--weight_iter_num', type=int, default='45')
    parser.add_argument('--SAVE_WEIGHTS', type=bool, default=True)
    args = parser.parse_args()
    main(args)