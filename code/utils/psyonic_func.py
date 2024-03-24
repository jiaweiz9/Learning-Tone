import sys, os, time
import numpy as np
sys.path.append(os.path.dirname((os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from utils.class_motionhelper import Timer

# Calculation Functions - Obeservations
def get_velocity(prev_action, curr_action, ros_rate=50):
    vel = (curr_action - prev_action) / (1 / ros_rate) # 50HZ, ros_rate
    return vel

# Clip the action based on the velocity limit
def vel_clip_action(prev_action, action, min_vel=-1.0, max_vel=1.0, ros_rate=50):
    vel = get_velocity(prev_action * (3.14/180), action * (3.14/180))
    vel_clip = np.clip(vel, min_vel, max_vel)
    print("vel_clip: ", vel_clip)
    assert vel_clip.all() >= min_vel
    assert vel_clip.all() <= max_vel
    delta_action = vel_clip * (180 / 3.14) * (1 / (ros_rate))
    curr_action = prev_action + delta_action
    return curr_action, vel_clip


# Calculate the acceleration based on current velocity and previous velocity
def get_acceleration(prev_vel, curr_vel, ros_rate=50):
    acc = (curr_vel - prev_vel) / (1 / ros_rate)
    return acc


# TODO: Map policy output numbers to the range(init_pose ~ max_pose), instead of clipping
def beta_dist_to_action_space(action, action_min, action_max):
    action = action * (action_max - action_min) + action_min
    return action

# TODO: Clip the action based on acceleration
def acc_clip_action():
    pass