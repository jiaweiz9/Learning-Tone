from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
from numpy.typing import NDArray
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from typing import List, Any, Dict, Literal
import scipy
from scipy.fft import fft, fftfreq, fftshift
# from utils.audio_transform import pre_emphasis_high_pass, waveform_to_frequence
import python_speech_features


class RewardUtils:
    def __init__(self,
                 ref_audio: NDArray[Any]=None, 
                 episode_length: int=100,
                 sr: int=44100,
                 iteration: int=0):
        self.rec_audio = None
        self.ref_audio = ref_audio
        self.episode_length = episode_length
        self.sr = sr
        self.iteration = iteration

        self.__check_audio_data(self.ref_audio)
        self.__generate_ref_audio_info()


    def __check_audio_data(self, audio):
        # make sure the reference audio, recorded audio and episode length are set
        if audio is None:
            raise ValueError("Audio must be set")
        
        # make sure the audio data is numpy array
        if isinstance(audio, np.ndarray) is False:
            raise ValueError("Audio must be numpy arrays, but is {}.".format(type(audio)))
        
    def __generate_ref_audio_info(self):
        self._ref_onset_strength_envelop = librosa.onset.onset_strength(y=self.ref_audio, sr=self.sr)
        self._ref_onset_strength_envelop[self._ref_onset_strength_envelop < 10] = 0

        self._ref_hitting_timings = librosa.onset.onset_detect(onset_envelope=self._ref_onset_strength_envelop, sr=self.sr, units='time', normalize=True)
        self._ref_hitting_frames = (self._ref_hitting_timings * self.sr).astype(int)

        self.ref_max_amp = np.max(abs(self.ref_audio))


    def load_rec_audio(self, rec_audio):
        self.rec_audio = rec_audio
        self.__check_audio_data(self.rec_audio)
        self.__generate_rec_audio_info()
        

    def __generate_rec_audio_info(self):
        self._rec_onset_strength_envelop = librosa.onset.onset_strength(y=self.rec_audio, sr=self.sr)
        self._rec_onset_strength_envelop[self._rec_onset_strength_envelop < 10] = 0

        # Get the hitting timings from the onset strength envelop
        self._rec_hitting_timings = librosa.onset.onset_detect(onset_envelope=self._rec_onset_strength_envelop, sr=self.sr, units='time', normalize=True)
        self._rec_hitting_frames = (self._rec_hitting_timings * self.sr).astype(int)

        self.rec_max_amp = np.max(abs(self.rec_audio))


    def amplitude_reward(self, amp_scale=1e2):
        max_amp_diff = np.abs(self.rec_max_amp - self.ref_max_amp)
        # max_amp = max(self.rec_max_amp, self.ref_max_amp)
        # return np.exp(-max_amp_diff * 10)
        return 1 - max_amp_diff / 0.13 if max_amp_diff < 0.13 else 0
    
    def double_hit_amp_reward(self):
        assert len(self._rec_hitting_frames) == 2 and len(self._ref_hitting_frames) == 2
        rec_hit_mid_frame = (self._rec_hitting_frames[0] + self._rec_hitting_frames[1]) / 2
        ref_hit_mid_frame = (self._ref_hitting_frames[0] + self._ref_hitting_frames[1]) / 2
        rec_amp_1 = np.max(abs(self.rec_audio[:rec_hit_mid_frame]))
        rec_amp_2 = np.max(abs(self.rec_audio[rec_hit_mid_frame:]))
        print(f"recorded audio hitting amplitude: {[rec_amp_1, rec_amp_2]}")

        rec_amp_1 = np.max(abs(self.ref_audio[:rec_hit_mid_frame]))
        rec_amp_2 = np.max(abs(self.ref_audio[rec_hit_mid_frame:]))
        print(f"reference audio hitting amplitude: {[rec_amp_1, rec_amp_2]}")



    def hitting_times_reward(self) -> float:
        # # if no hit, return -10
        if len(self._rec_hitting_timings) == 0:
            return -5
        # the returned difference will not be smaller than -20
        else:
            return -min(abs(len(self._rec_hitting_timings) - len(self._ref_hitting_timings)), 20)
        # return -min(abs(len(self._rec_hitting_timings) - len(self._ref_hitting_timings)), 20)
        

    # Compute the DTW distance (Dynamic Time Warping) between the onset strength envelops of the recorded and reference audio, serving as a measure of shape similarity
    def onset_shape_reward(self) -> float:
        diff = self.__mel_filterbank()
        print(diff)
        shape_reward = -min(diff / 1000, 20)
        # print(f"Shape diff: {diff}")
        return shape_reward
    
    def __mel_filterbank(self):
        # mcc_feat = python_speech_features.mfcc(self.rec_audio, self.sr)
        fbank_feat = python_speech_features.logfbank(self.rec_audio[:88200] / self.rec_max_amp, self.sr)

        # mcc_feat_ref = python_speech_features.mfcc(self.ref_audio, self.sr)
        fbank_feat_ref = python_speech_features.logfbank(self.ref_audio[:88200] / self.ref_max_amp, self.sr)

        print(len(fbank_feat))
        print(len(fbank_feat_ref))

        diff, _ = fastdtw(fbank_feat, fbank_feat_ref, radius=5)

        return diff

    def __compute_freqs_diffs(self, ref_audio, rec_audio):
        assert ref_audio.shape == rec_audio.shape, "To compute frequencies difference, ref and rec should have the same shape!"

        rec_S = librosa.stft(rec_audio / np.max(rec_audio))
        rec_S_db = librosa.amplitude_to_db(np.abs(rec_S), ref=np.max)
        ref_S = librosa.stft(ref_audio / np.max(ref_audio))
        ref_S_db = librosa.amplitude_to_db(np.abs(ref_S), ref=np.max)

        # rec_freqs = librosa.fft_frequencies()
        return np.abs(rec_S_db - ref_S_db)

    def hitting_timing_reward(self) -> float:
        if len(self._rec_hitting_frames) == len(self._ref_hitting_frames):
            # time_diff = abs(np.sum(onset_hit_times_ref - onset_hit_times_rec))
            timing_diff = abs(np.sum(self._rec_hitting_timings - self._ref_hitting_timings))
            # return 1 - 4 * timing_diff ** 2 if timing_diff < 0.5 else 0
            return 1 - timing_diff / 2 if timing_diff / 2 < 1 else 0
        return 0
    
    def success_threshold_scheduler(self):
        '''
        This function should return a value between 0 and 1, which will be used to determine the success threshold
        '''
        timing_threshold = min(0.8 + 0.002 * self.iteration, 0.9)
        amplitude_threshold = min(0.8 + 0.002 * self.iteration, 0.9)
        return timing_threshold, amplitude_threshold
    

    def success_reward(self) -> float:
        '''
        Give this reward only when hitting the desired times, with good timing, and shape
        '''
        timing_threshold, amplitude_threshold = self.success_threshold_scheduler()
        # print(f"timing threshold: {timing_threshold}, amplitude threshold: {amplitude_threshold}")
        return 100 if (
            len(self._rec_hitting_timings) == len(self._ref_hitting_timings) and
            self.amplitude_reward() > 0.3 and
            self.onset_shape_reward() > -10 and
            self.hitting_timing_reward() > 0.9     # this means the timing error is smaller than 0.5 seconds
        ) else 0
    

    def step_amp_reward(self, prev_step_rec_audio, step_rec_audio, cur_step, onset_threshold=0.1, frame_range=4410):
        '''
            New step reward: 
            if current step amplitude is louder enough than the previous one (considered as one hitting), 
        '''
        if np.max(abs(step_rec_audio)) - np.max(abs(prev_step_rec_audio)) > onset_threshold:
            for hitting_frame in self._ref_hitting_frames:
                if hitting_frame - frame_range <= cur_step * len(step_rec_audio) <= hitting_frame + frame_range:
                    return 0.0
            return -1.0
        else:
            return 0.0





def __visualize_audio_step(ref_audio, rec_audio, rec_idx, step_rew, sr=44100) -> None:
        import matplotlib.pyplot as plt
        import os

        plt.figure(figsize=(20, 6))
        max_len = max(len(rec_audio), len(ref_audio))
        ref_audio = np.pad(ref_audio, (0, max_len - len(ref_audio)), 'constant')
        rec_audio = np.pad(rec_audio, (0, max_len - len(rec_audio)), 'constant')

        time = np.arange(0, max_len) / sr

        plt.plot(time, rec_audio, color='blue', alpha=0.3)
        plt.plot(time, ref_audio, color='red', alpha=0.3)

        rec_idx = rec_idx * 0.025
        plt.scatter(rec_idx, step_rew, color='black', marker='x')

        # plt.title(f'Episode {rec_idx} - Reference Audio (red) vs. Recorded Audio (blue)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend(['Recorded Audio', 'Reference Audio', 'Step Reward'])

        # file_name = f"episode_{self.num_timesteps}.png"
        # img_path = os.path.join(self.figures_path, file_name)
        # plt.savefig(img_path)
        plt.show()
        plt.close()

        # if not os.path.exists(f"results/audios/{self.folder_name}"):
        #     os.makedirs(f"results/audios/{self.folder_name}")
        # wavio.write(f"results/audios/{self.folder_name}/episode_{self.num_timesteps}.wav", rec_audio[:88200], rate=44100, sampwidth=4)








if __name__ == "__main__":
    ref_audio, _ = librosa.load("ref_audio/xylophone_keyB/amp06_013.wav", sr=44100)

    reward_utils = RewardUtils(
        ref_audio=ref_audio,
        episode_length=50,
        sr=44100,
    )

    rec_audio, _ = librosa.load("results/audios/0628_1919-97ukixlk/episode_4000.wav", sr=44100)

    rec_idx = np.arange(50)
    rec_step_rews = []
    prev_step_rec_audio = step_rec_audio = 0
    for i in rec_idx:
        prev_step_rec_audio = step_rec_audio
        step_rec_audio = rec_audio[i * 1100 : (i+1) * 1100]
        step_rew = reward_utils.step_amp_reward(
            prev_setp_rec_audio=prev_step_rec_audio,
            step_rec_audio=step_rec_audio,
            cur_step=i,
            onset_threshold=0.1,
            frame_range=2205
        )
        rec_step_rews.append(step_rew)
    
    print(rec_idx)
    print(rec_step_rews)
    __visualize_audio_step(
        ref_audio=ref_audio,
        rec_audio=rec_audio,
        rec_idx=rec_idx,
        step_rew=rec_step_rews,
        sr=44100
    )