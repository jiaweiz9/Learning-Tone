import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def amplitude_reward(audio_data_step_window, ref_data_step_window, amp_scale=1e2):
    mean_amp = np.mean(abs(audio_data_step_window))
    if np.isnan(mean_amp):
                mean_amp = 0
    mean_ref_amp = np.mean(abs(ref_data_step_window))
    gap_amp = np.abs(mean_amp - mean_ref_amp)
    amp_reward = np.exp(-gap_amp)
    return amp_reward, mean_amp


def onset_rewards(audio_data, ref_audio, ref_sr):

    # Normalize audio data to 0-1
    norm_audio_rec = audio_data / np.max(audio_data)
    norm_audio_ref = ref_audio / np.max(ref_audio)

    # Reference sound Onset
    onset_envelop_ref = librosa.onset.onset_strength(y=norm_audio_ref, sr=ref_sr)
    norm_onset_envelop_ref = onset_envelop_ref / np.max(onset_envelop_ref) # Normalize onset strength to 0-1
    onset_times_ref = librosa.onset.onset_detect(y=norm_audio_ref, sr=ref_sr, onset_envelope=onset_envelop_ref, units='time')
    beat_cnt_ref = onset_times_ref.size

    # Generated sound Onset
    onset_envelop_rec = librosa.onset.onset_strength(y=norm_audio_rec, sr=ref_sr)
    norm_onset_envelop_rec = onset_envelop_rec / np.max(onset_envelop_rec) # Normalize onset strength to 0-1
    onset_times_rec = librosa.onset.onset_detect(y=norm_audio_rec, sr=ref_sr, onset_envelope=onset_envelop_rec, units='time')
    beat_cnt_rec = onset_times_rec.size

    print("beat_cnt_ref: ", beat_cnt_ref)
    print("beat_cnt_rec: ", beat_cnt_rec)

    # Eq(2). Onset strength reward
    dtw_onset, _ = fastdtw(norm_onset_envelop_rec, norm_onset_envelop_ref) # Onset DTW
    onset_reward = (-dtw_onset)

    # Eq(3). Onset timing reward
    max_len = max(len(onset_times_ref), len(onset_times_rec))
    onset_times_ref = np.concatenate((onset_times_ref, np.zeros(max_len - len(onset_times_ref))))
    onset_times_rec = np.concatenate((onset_times_rec, np.zeros(max_len - len(onset_times_rec))))
    timing_reward = np.exp(-euclidean(onset_times_ref, onset_times_rec))

    # Eq(4). Hit reward
    hit_reward = -(beat_cnt_rec - beat_cnt_ref) ** 2

    # assert False, "Check the reward values"
    return onset_reward, timing_reward, hit_reward

def dtw_reward(ref_audio_envelop, rec_audio_envelop):
    # onset_envelop_ref = librosa.onset.onset_strength(y=ref_data_step_window, sr=44100)
    # print(onset_envelop_ref)
    # norm_onset_envelop_ref = onset_envelop_ref / np.max(onset_envelop_ref) # Normalize onset strength to 0-1
    
    # onset_envelop_rec = librosa.onset.onset_strength(y=audio_data_step_window, sr=44100)
    # print(onset_envelop_rec)
    # norm_onset_envelop_rec = onset_envelop_rec / np.max(onset_envelop_rec) # Normalize onset strength to 0-1

    # print("shape of norm_onset_envelop_rec: ", norm_onset_envelop_rec.shape)
    # print("shape of norm_onset_envelop_ref: ", norm_onset_envelop_ref.shape)

    dtw_onset, _ = fastdtw(ref_audio_envelop, rec_audio_envelop) # Onset DTW
    dtw = (-dtw_onset)
    return dtw


def onset_timing_reward(audio_data_step_window, ref_data_step_window):
    pass

def onset_hit_reward(ref_audio, rec_audio, epi_length, effect_window=10, sr=44100):
    onset_strength_envelop_rec = librosa.onset.onset_strength(y=rec_audio, sr=sr)
    onset_strength_envelop_ref = librosa.onset.onset_strength(y=ref_audio, sr=sr)
    onset_strength_envelop_rec = np.array([0.0 if onset_strength < 5 else onset_strength for onset_strength in onset_strength_envelop_rec])
    onset_strength_envelop_ref = np.array([0.0 if onset_strength < 5 else onset_strength for onset_strength in onset_strength_envelop_ref])

    # print(np.max(np.abs(rec_audio)))
    onset_hit_times_rec = librosa.onset.onset_detect(y=rec_audio, onset_envelope=onset_strength_envelop_rec, sr=44100, units='time', normalize=True)

    onset_hit_times_ref = librosa.onset.onset_detect(y=ref_audio, onset_envelope=onset_strength_envelop_ref, sr=44100, units='time', normalize=True)
    
    onset_hit_times_ref = (onset_hit_times_ref * 44100).astype(int)
    onset_hit_times_rec = (onset_hit_times_rec * 44100).astype(int)
    
    print("onset_hit_times_ref: ", onset_hit_times_ref)
    print("onset_hit_times_rec: ", onset_hit_times_rec)

    hit_reward_list = np.zeros(epi_length)
    hit_reward_list[-1] = - min(abs(len(onset_hit_times_rec) - len(onset_hit_times_ref)), 10)

    if len(onset_hit_times_rec) == len(onset_hit_times_ref):
        hit_reward_list[-1] += np.exp(- 5 * euclidean(onset_hit_times_ref[0] / 44100.0, onset_hit_times_ref[0] / 44100.0)) * 10

    return hit_reward_list


def assign_rewards_to_episode(ref_audio, rec_audio, epi_length):

    time_step_window = rec_audio.shape[0] // epi_length
    amp_reward_list = []

    ref_audio_envelop = librosa.onset.onset_strength(y=ref_audio, sr=44100)
    print("length of ref_audio: ", len(ref_audio))

    rec_audio_envelop = librosa.onset.onset_strength(y=rec_audio, sr=44100)
    print("length of rec_audio: ", len(rec_audio))

    normalized_ref_audio = ref_audio / np.max(ref_audio)
    normalized_rec_audio = rec_audio / np.max(rec_audio)

    ref_audio_envelop = ref_audio_envelop / np.max(ref_audio_envelop)
    rec_audio_envelop = rec_audio_envelop / np.max(rec_audio_envelop)

    for i in range(epi_length):
        audio_data_step_window = normalized_rec_audio[i * time_step_window : (i+1) * time_step_window]
        ref_data_step_window = normalized_ref_audio[i * 884 : (i+1) * 884]
        amp_reward, mean_amp = amplitude_reward(audio_data_step_window, ref_data_step_window)

        amp_reward_list.append(amp_reward)
        # dtw_reward_list.append(dtw)
    
    onset_hit_reward_list = onset_hit_reward(ref_audio, rec_audio, epi_length)

    return np.array(amp_reward_list), np.array(onset_hit_reward_list)



def rewards_for_stages(ref_audio, rec_audio, epi_length):
    reward = 0
    onset_strength_envelop_rec = librosa.onset.onset_strength(y=rec_audio, sr=44100)
    onset_strength_envelop_ref = librosa.onset.onset_strength(y=ref_audio, sr=44100)
    onset_strength_envelop_rec = np.array([0.0 if onset_strength < 5 else onset_strength for onset_strength in onset_strength_envelop_rec])
    onset_strength_envelop_ref = np.array([0.0 if onset_strength < 5 else onset_strength for onset_strength in onset_strength_envelop_ref])

    onset_hit_times_rec = librosa.onset.onset_detect(y=rec_audio, onset_envelope=onset_strength_envelop_rec, sr=44100, units='time', normalize=True)
    onset_hit_times_ref = librosa.onset.onset_detect(y=ref_audio, onset_envelope=onset_strength_envelop_ref, sr=44100, units='time', normalize=True)
    # stage 1: correct hitting times when it is wrong
    if len(onset_hit_times_rec) != len(onset_hit_times_ref):
        reward = -(len(onset_hit_times_rec) - len(onset_hit_times_ref)) ** 2

    # stage 2: constrain hitting timing to be close
    else:
        reward -= np.exp(euclidean(onset_hit_times_rec, onset_hit_times_ref))

    return reward