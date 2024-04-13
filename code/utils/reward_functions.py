import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def amplitude_reward(audio_data_step_window, ref_data_step_window, amp_scale=1e2):
    mean_amp = np.mean(abs(audio_data_step_window))* amp_scale
    if np.isnan(mean_amp):
                mean_amp = 0
    mean_ref_amp = np.mean(abs(ref_data_step_window))* amp_scale
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


def onset_hit_reward(ref_audio, rec_audio, epi_length, effect_window=15):
    """Find when hitting happens in all audio data step window s, and compare with the conresponding ref data step window. 

    Args:
        ref_audio (_type_): _description_
        rec_audio (_type_): _description_
        epi_length (_type_): _description_

    Returns:
        hit_reward_list (_type_): _description_
    """
    hit_reward_list = []
    time_step_window = rec_audio.shape[0] // epi_length
    
    # ref_audio = ref_audio / np.max(ref_audio)
    # rec_audio = rec_audio / np.max(rec_audio)
    if np.max(np.abs(rec_audio)) < 0.05:
        onset_hit_times_rec = np.array([])
    else:
        onset_hit_times_rec = librosa.onset.onset_detect(y=rec_audio, sr=44100, units='time')

    onset_hit_times_ref = librosa.onset.onset_detect(y=ref_audio, sr=44100, units='time')
    
    onset_hit_times_ref = (onset_hit_times_ref * 44100).astype(int)
    onset_hit_times_rec = (onset_hit_times_rec * 44100).astype(int)
    
    print("onset_hit_times_ref: ", onset_hit_times_ref)
    print("onset_hit_times_rec: ", onset_hit_times_rec)
    # hit_time_ref_set = set(onset_hit_times_ref)
    # hit_time_rec_set = set(onset_hit_times_rec)
    # print(hit_time_rec_set)
    # print(hit_time_ref_set)

    # for i in range(epi_length):
    #     step_window_set_ref = set(range(i * 884, (i + 1) * 884))
    #     step_window_set_rec = set(range(i * time_step_window, (i+1) * time_step_window))

    #     hit_count_ref = len(step_window_set_ref.intersection(hit_time_ref_set))
    #     hit_count_rec = len(step_window_set_rec.intersection(hit_time_rec_set))

    #     hit_reward = -(hit_count_rec - hit_count_ref) ** 2
    #     hit_reward_list.append(hit_reward)
    hit_time_indexes = [hit_time // time_step_window for hit_time in onset_hit_times_rec]
    ref_hit_time_indexes = [hit_time // 882 for hit_time in onset_hit_times_ref]
    print('hit_time_indexes: ', hit_time_indexes)
    print('ref_hit_time_indexes: ', ref_hit_time_indexes)

    hit_reward_list = np.zeros(epi_length)

    for hit_time_index in hit_time_indexes:
        hit_reward_list[hit_time_index] = -1
        
        for ref_hit_time_index in ref_hit_time_indexes:
            if abs(hit_time_index - ref_hit_time_index) < effect_window:
                hit_reward_list[hit_time_index] = 1
                break
            
    
    # if len(hit_time_rec_set) != len(hit_time_ref_set):
    #     timing_reward = 0
    # else:
    #     timing_reward = np.exp(-euclidean(onset_hit_times_ref / np.linalg.norm(onset_hit_times_ref), onset_hit_times_rec / np.linalg.norm(onset_hit_times_rec)))

    return hit_reward_list


def assign_rewards_to_episode(ref_audio, rec_audio, epi_length):

    time_step_window = rec_audio.shape[0] // epi_length
    amp_reward_list = []
    dtw_reward_list = []

    ref_audio_envelop = librosa.onset.onset_strength(y=ref_audio, sr=44100)
    print("length of ref_audio: ", len(ref_audio))

    rec_audio_envelop = librosa.onset.onset_strength(y=rec_audio, sr=44100)
    print("length of rec_audio: ", len(rec_audio))

    for i in range(epi_length):
        audio_data_step_window = rec_audio[i * time_step_window : (i+1) * time_step_window]
        ref_data_step_window = ref_audio[i * 884 : (i+1) * 884]
        amp_reward, mean_amp = amplitude_reward(audio_data_step_window, ref_data_step_window)
        # dtw = dtw_reward(audio_data_step_window, ref_data_step_window)

        amp_reward_list.append(amp_reward)
        # dtw_reward_list.append(dtw)
    
    onset_hit_reward_list = onset_hit_reward(ref_audio, rec_audio, epi_length)
    # print("Hit Times Reward:", hit_times_reward)
    # print("Hit Timing Reward:", hit_timing_reward)

    return np.array(amp_reward_list), np.array(onset_hit_reward_list), dtw_reward(ref_audio_envelop, rec_audio_envelop)