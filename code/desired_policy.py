from ros_func.publisher import QPosPublisher
from utils.sound_recorder import SoundRecorder
import argparse, os, sys, wandb, copy, rospy, time
from utils.eval_result import visualize_audio
import librosa
import numpy as np
import wavio
from utils.reward_functions import assign_rewards_to_episode, move_penalty, dtw_reward

initial_pose = np.array([50, 70, 110, 115, 50, -10])
pose_upper = np.array([-10])
pose_lower = np.array([-50])

desired_high_actions = np.zeros((100,))
desired_mid_actions = np.zeros((100,))
desired_low_actions = np.zeros((100,))
desired_high_actions.fill(-10)
desired_mid_actions.fill(-10)
desired_low_actions.fill(-10)
# ===================desired high movement (0.5)=========================
desired_high_actions[20:23] = -50

# ===================desired mid movement==========================
desired_mid_actions[20:25] = -25
desired_mid_actions[25:30] = -50

# ===================desired low movement===========================
desired_low_actions[15:27] = -35
desired_low_actions[27:30] = -50

# ===================shaking movement=========================
# desired_actions[0::2] = -50
# desired_actions[1::2] = -10

# ===================only motor noise=========================
# desired_actions[0::2] = -30
# desired_actions[1::2] = -20

# ===================hit at the end===========================
# desired_actions[99] = -50
# print(desired_actions)

rospy.init_node('psyonic_for_real', anonymous=True)
# QPosPublisher.publish_once(initial_pose)
qpospublisher = QPosPublisher()
# qpospublisher.publish_once(initial_pose)

parser = argparse.ArgumentParser()
parser.add_argument("--desired_force", type=str, default='high', help="high|mid|low")
args = parser.parse_args()

if args.desired_force == "high":
    desired_actions = desired_high_actions
elif args.desired_force == "mid":
    desired_actions = desired_mid_actions
else:
    desired_actions = desired_low_actions

Recorder = SoundRecorder()
Recorder.start_recording()
for i in range(100):
    pose_cmd = np.concatenate((initial_pose[:-1], [desired_actions[i]]), axis=0)
    print(pose_cmd)
    qpospublisher.publish_once(pose_cmd)

time.sleep(0.1) # sleep at the end to take care of hitting at the end of the episode
Recorder.stop_recording()
rec_audio = Recorder.get_current_buffer()
wavio.write(f"ref_audio/ref_{args.desired_force}.wav", rec_audio, rate=44100, sampwidth=4)

ref_audio, ref_sr = librosa.load("ref_audio/xylophone/ref_hit1_clip.wav", sr=44100) # load reference audio
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