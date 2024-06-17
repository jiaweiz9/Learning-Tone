import argparse, os, sys, wandb, copy, rospy, time
from psyonic_playing_xylophone.ros_interfaces.publisher import QPosPublisher
from psyonic_playing_xylophone.utils.sound_recorder import visualize_audio, SoundRecorder
import librosa
import numpy as np
import wavio
# from utils.reward_functions import assign_rewards_to_episode, move_penalty, dtw_reward

initial_pose = np.array([50, 70, 110, 115, 50, -20])
pose_upper = np.array([-10])
pose_lower = np.array([-50])

desired_high_actions = np.zeros((100,))
desired_mid_actions = np.zeros((100,))
desired_low_actions = np.zeros((100,))

desired_high_actions.fill(-10)
desired_mid_actions.fill(-10)
desired_low_actions.fill(-10)

desired_actions = np.zeros((50,))
desired_actions.fill(-20)
desired_actions[20:22] = -10
desired_actions[22:24] = -30
desired_actions[24:26] = -50

rospy.init_node('psyonic_for_real', anonymous=True)
# QPosPublisher.publish_once(initial_pose)
qpospublisher = QPosPublisher()
# qpospublisher.publish_once(initial_pose)

Recorder = SoundRecorder()
Recorder.start_recording()
for i in range(50):
    pose_cmd = np.concatenate((initial_pose[:-1], [desired_actions[i]]), axis=0)
    print(pose_cmd)
    qpospublisher.publish_once(pose_cmd)

time.sleep(0.1) # sleep at the end to take care of hitting at the end of the episode
Recorder.stop_recording()
rec_audio = Recorder.get_episode_audio()
# wavio.write(f"ref_audio/ref_{args.desired_force}.wav", rec_audio, rate=44100, sampwidth=4)

# ref_audio, ref_sr = librosa.load("ref_audio/xylophone/ref_hit1_clip.wav", sr=44100) # load reference audio

visualize_audio(data=rec_audio, sr=44100)