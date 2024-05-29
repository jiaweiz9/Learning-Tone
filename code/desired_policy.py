from ros_func.publisher import QPosPublisher
from utils.sound_recorder import SoundRecorder
import argparse, os, sys, wandb, copy, rospy, time
from utils.eval_result import visualize_audio
import librosa
import numpy as np
from utils.reward_functions import assign_rewards_to_episode, move_penalty, dtw_reward

initial_pose = np.array([50, 70, 110, 115, 50, -10])
pose_upper = np.array([-10])
pose_lower = np.array([-50])

desired_actions = np.zeros((100,))
desired_actions.fill(-10)
# ===================desired movement=========================
# desired_actions[20:30] = -50

# ===================shaking movement=========================
# desired_actions[0::2] = -50
# desired_actions[1::2] = -10

# ===================only motor noise=========================
desired_actions[0::2] = -30
desired_actions[1::2] = -20

# ===================hit at the end===========================
# desired_actions[99] = -50
# print(desired_actions)

rospy.init_node('psyonic_for_real', anonymous=True)
# QPosPublisher.publish_once(initial_pose)
qpospublisher = QPosPublisher()
# qpospublisher.publish_once(initial_pose)

Recorder = SoundRecorder()
Recorder.start_recording()
for i in range(100):
    pose_cmd = np.concatenate((initial_pose[:-1], [desired_actions[i]]), axis=0)
    print(pose_cmd)
    qpospublisher.publish_once(pose_cmd)

time.sleep(0.1) # sleep at the end to take care of hitting at the end of the episode
Recorder.stop_recording()
rec_audio = Recorder.get_current_buffer()

ref_audio, ref_sr = librosa.load("ref_audio/xylophone/ref_hit1_clip.wav", sr=44100) # load reference audio

rec_audio[abs(rec_audio) < 0.04] = 0
amp_reward_list, hit_reward_list, timing_reward_list = assign_rewards_to_episode(ref_audio, 
                                                                                 rec_audio.squeeze(), 
                                                                                 100)

dtw_rwd = dtw_reward(ref_audio, rec_audio.squeeze())

move_penalty_list = move_penalty(desired_actions)
# D = np.abs(librosa.stft(ref_audio))
# times = librosa.times_like(D, sr=44100)
# print(D.shape)
# print(times)
print("hit_rew ", np.sum(hit_reward_list))
print("timinf_rew ", np.sum(timing_reward_list))
print("move_rew ", np.sum(move_penalty_list))
print("dtw_rew: ", dtw_rwd)
visualize_audio(ref_audio=ref_audio, audio_data=rec_audio, sr=ref_sr, block=True)