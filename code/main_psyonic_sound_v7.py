import argparse, os, sys
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from psyonic_real_v7 import *

def main(args):
    Psyonic = PsyonicForReal(args)
    Psyonic.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument('--min_vel', type=float, default=-5.0)
    parser.add_argument('--max_vel', type=float, default=5.0)
    parser.add_argument('--velocity_free_coef', type=float, default=1.1)
    parser.add_argument('--ref_audio_path', type=str, default='ref_audio/xylophone/ref_hit2_filtered.wav')

    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--ros_rate', type=int, default=50)
    parser.add_argument('--record_duration', type=int, default=4)
    parser.add_argument('--mini_batch_size', type=int, default=100)
    
    parser.add_argument('--n_epi', type=int, default=1) # n_epi * 50 * record_duration = steps per sampling
    parser.add_argument('--k_epoch', type=int, default=10) # num of epoch for gradient descent
    parser.add_argument('--max_pos', type=float, default=1.0)
    parser.add_argument('--obs_dim', type=int, default=3)
    parser.add_argument('--act_dim', type=int, default=1)
    parser.add_argument('--beta_dist', action="store_true")
    parser.add_argument('--reload_iter', type=int, default=0)
    
    parser.add_argument('--h_dims', nargs="+", type=int, default=[128, 128]) # change this term
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lmbda', type=float, default=0.95)

    parser.add_argument('--lr_actorcritic', type=float, default=1e-5) # change this term 0.003 ~ 5e-6
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--max_grad', type=float, default=0.5)
    parser.add_argument('--samplerate', type=int, default=44100) # Audio sample rate
    parser.add_argument('--WANDB', action="store_true")
    parser.add_argument('--weight_iter_num', type=int, default='45')
    parser.add_argument('--SAVE_WEIGHTS', action="store_true")
    args = parser.parse_args()
    main(args)