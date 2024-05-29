import sys, os, time
import numpy as np
sys.path.append(os.path.dirname((os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
# from utils.class_motionhelper import Timer

# Calculation Functions - Obeservations
def get_velocity(prev_action, curr_action, ros_rate=50):
    vel = (curr_action - prev_action) / (1 / ros_rate) # 50HZ, ros_rate
    return vel

# Clip the action based on the velocity limit
def vel_clip_action(prev_action, action, min_vel=-5.0, max_vel=5.0, ros_rate=50):
    
    vel = get_velocity(prev_action, action)
    vel_clip = np.clip(vel, min_vel, max_vel)
    assert vel_clip.all() >= min_vel
    assert vel_clip.all() <= max_vel
    delta_action = vel_clip * (1 / (ros_rate))
    curr_action = prev_action + delta_action
    # print("action_after_clip_in: ", curr_action)
    return curr_action, vel_clip

def clip_max_move(prev_action, curr_action, max_move):
    if curr_action - prev_action > max_move:
        curr_action = prev_action + max_move
    elif prev_action - curr_action > max_move:
        curr_action = prev_action - max_move
    return curr_action

# Calculate the acceleration based on current velocity and previous velocity
def get_acceleration(prev_vel, curr_vel, ros_rate=50):
    acc = (curr_vel - prev_vel) / (1 / ros_rate)
    return acc


def beta_dist_to_action_space(beta_out, action_min, action_max):
    action = beta_out * (action_max - action_min) + action_min
    return action

def action_space_to_beta_dist(action, action_min, action_max):
    beta_x = (action - action_min) / (action_max - action_min)
    return beta_x

def action_space_to_norm(action, action_min=-50, action_max=-10):
    norm_x = 2 * (action - action_min) / (action_max - action_min) - 1
    return norm_x

def norm_to_action_space(norm_x, action_min=-50, action_max=-10):
    action = (norm_x + 1) * (action_max - action_min) / 2 + action_min
    return action

def normilize_obs(obs, total_timestep = 100.0, min_pos = -50.0, max_pos = -10.0):
    # Normalize the time step
    # obs[0] = (obs[0] / total_timestep) * 2 - 1
    
    # Normalize the action space
    obs = (obs - min_pos) / (max_pos - min_pos) * 2 - 1

    return obs


def acc_clip_action():
    pass