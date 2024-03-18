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
    # print("shape of audio_data", audio_data.shape)
    # print("shape of ref_audio", ref_audio.shape)
    # print("ref_sr", ref_sr)

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

    print("onsert_times_ref", onset_times_ref)
    print("onset_times", onset_times_rec)

    # Eq(2). Onset strength reward
    dtw_onset, _ = fastdtw(norm_onset_envelop_rec, norm_onset_envelop_ref) # Onset DTW
    onset_reward = (-dtw_onset)

    # Eq(3). Onset timing reward
    max_len = max(len(onset_times_ref), len(onset_times_rec))
    onset_times_ref = np.concatenate((onset_times_ref, np.zeros(max_len - len(onset_times_ref))))
    onset_times_rec = np.concatenate((onset_times_rec, np.zeros(max_len - len(onset_times_rec))))
    timing_reward = np.exp(-euclidean(onset_times_ref, onset_times_rec))

    # Eq(4). Hit reward
    if beat_cnt_ref == beat_cnt_rec:
        hit_reward = beat_cnt_rec
    else:
        hit_reward = 0

    assert False, "Check the reward values"
    return onset_reward, hit_reward, timing_reward