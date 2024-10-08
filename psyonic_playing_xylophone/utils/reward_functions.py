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


class RecRefRewardFunction:
    def __init__(self, rec_audio:NDArray[Any]=None, 
                 ref_audio: NDArray[Any]=None, 
                 episode_length: int=100,
                 sr: int=44100,
                 iteration: int=0):
        self.rec_audio = rec_audio
        self.ref_audio = ref_audio
        self.episode_length = episode_length
        self.sr = sr
        self.iteration = iteration

        self.__check_audio_data()
        self.__generate_audio_strength_info()


    def __check_audio_data(self):
        # make sure the reference audio, recorded audio and episode length are set
        if self.rec_audio is None \
            or self.ref_audio is None:
            raise ValueError("Reference audio, recorded audio must be set")
        
        # make sure the audio data is numpy array
        if isinstance(self.rec_audio, np.ndarray) is False \
            or isinstance(self.ref_audio, np.ndarray) is False:
            raise ValueError("Reference audio and recorded audio must be numpy arrays, but reference audio is {} and recorded audio is {}".format(type(self.ref_audio), type(self.rec_audio)))
        
        # remove motor noise whose amplitude is around 0.02
        # hit_amp_th = 0.02
        self.rec_max_amp = np.max(abs(self.rec_audio))
        # if self.rec_max_amp < hit_amp_th:
        #     self.rec_audio[:] = 1e-5

        # data_fft = np.fft.fft(self.rec_audio)
        # freqs = np.fft.fftfreq(len(data_fft), 1 / 44100)
        # data_fft[np.abs(freqs) < 1000] = 0
        # self.rec_audio = np.real(np.fft.ifft(data_fft))

        self.ref_max_amp = np.max(self.ref_audio)
        print("ref_audio shape: ", self.ref_audio.shape)
        print("rec_audio shape: ", self.rec_audio.shape)
        # print("ref max amp:", self.ref_max_amp)
        # print("rec max amp:", self.rec_max_amp)


    def __generate_audio_strength_info(self):
        self._rec_onset_strength_envelop = librosa.onset.onset_strength(y=self.rec_audio, sr=self.sr)
        self._ref_onset_strength_envelop = librosa.onset.onset_strength(y=self.ref_audio, sr=self.sr)

        # Normalize the onset strength envelop (done by onset_detect function with normalize=True)
        # self._rec_onset_strength_envelop /= np.max(self._rec_onset_strength_envelop)
        # self._ref_onset_strength_envelop /= np.max(self._ref_onset_strength_envelop)

        # Filter out the small onset strength values
        self._rec_onset_strength_envelop[self._rec_onset_strength_envelop < 10] = 0
        self._ref_onset_strength_envelop[self._ref_onset_strength_envelop < 10] = 0

        # Get the hitting timings from the onset strength envelop
        self._rec_hitting_timings = librosa.onset.onset_detect(onset_envelope=self._rec_onset_strength_envelop, sr=self.sr, units='time', normalize=True)

        self._ref_hitting_timings = librosa.onset.onset_detect(onset_envelope=self._ref_onset_strength_envelop, sr=self.sr, units='time', normalize=True)

        self._rec_hitting_frames = (self._rec_hitting_timings * self.sr).astype(int)
        self._ref_hitting_frames = (self._ref_hitting_timings * self.sr).astype(int)

        # if no hit, return 0
        print(f"ref hitting: {self._ref_hitting_timings}")
        print(f"rec hitting: {self._rec_hitting_timings}")
        # print(f"hitting frames reference {self._ref_hitting_frames}")
        # print(f"hitting frames recorded {self._rec_hitting_frames}")


    def amplitude_reward(self, amp_scale=1e2):
        max_amp_diff = np.abs(self.rec_max_amp - self.ref_max_amp)
        # max_amp = max(self.rec_max_amp, self.ref_max_amp)
        # return np.exp(-max_amp_diff * 10)
        return 1 - max_amp_diff / self.ref_max_amp if max_amp_diff < self.ref_max_amp else 0
    
    def double_hit_amp_reward(self):
        assert len(self._ref_hitting_frames) == 2
        if len(self._rec_hitting_frames) >= 2:
            print(self._rec_hitting_frames[0])
            rec_hit_mid_frame = (self._rec_hitting_frames[0] + self._rec_hitting_frames[1]) // 2
            ref_hit_mid_frame = (self._ref_hitting_frames[0] + self._ref_hitting_frames[1]) // 2
            print(rec_hit_mid_frame)
            print(ref_hit_mid_frame)
            rec_amp_1 = np.max(abs(self.rec_audio[:rec_hit_mid_frame]))
            rec_amp_2 = np.max(abs(self.rec_audio[rec_hit_mid_frame:]))
            print(f"recorded audio hitting amplitude: {[rec_amp_1, rec_amp_2]}")

            ref_amp_1 = np.max(abs(self.ref_audio[:ref_hit_mid_frame]))
            ref_amp_2 = np.max(abs(self.ref_audio[ref_hit_mid_frame:]))
            print(f"reference audio hitting amplitude: {[ref_amp_1, ref_amp_2]}")
            
            amp_diff = np.abs(rec_amp_1 - ref_amp_1) + np.abs(rec_amp_2 - ref_amp_2)
            return 1 - amp_diff / 0.3 if amp_diff < 0.3 else 0
        else:
            return 0



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
        shape_reward = -min(diff / 1500, 20)
        # print(f"Shape diff: {diff}")
        return shape_reward
    
    def __mel_filterbank(self):
        # mcc_feat = python_speech_features.mfcc(self.rec_audio, self.sr)
        fbank_feat = python_speech_features.logfbank(self.rec_audio[:88200] / self.rec_max_amp, self.sr)

        # mcc_feat_ref = python_speech_features.mfcc(self.ref_audio, self.sr)
        fbank_feat_ref = python_speech_features.logfbank(self.ref_audio[:88200] / self.ref_max_amp, self.sr)

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
            return 1 - timing_diff / 1 if timing_diff / 1 < 1 else 0
        return 0
    
    def success_threshold_scheduler(self):
        '''
        This function should return a value between 0 and 1, which will be used to determine the success threshold
        '''
        timing_threshold = min(0.85 + 0.01 * self.iteration // 10, 0.93)
        amplitude_threshold = min(0.85 + 0.01 * self.iteration // 10, 0.93)
        # timing_threshold = 0.8 if self.iteration < 50 else 0.9
        # amplitude_threshold = 0.8 if self.iteration < 50 else 0.9
        return timing_threshold, amplitude_threshold
    

    def success_reward(self) -> float:
        '''
        Give this reward only when hitting the desired times, with good timing, and shape
        '''
        # print("iteration in reward function:", self.iteration)
        timing_threshold, amplitude_threshold = self.success_threshold_scheduler()
        # print(f"timing threshold: {timing_threshold}, amplitude threshold: {amplitude_threshold}")
        return 100 if (
            len(self._rec_hitting_timings) == len(self._ref_hitting_timings) and
            self.amplitude_reward() > 0.3 and
            self.onset_shape_reward() > -15 and
            self.hitting_timing_reward() > 0.8     # this means the timing error is smaller than 0.2 seconds
        ) else 0
    
    def double_hit_success_reward(self):
        return 100 if (
            len(self._rec_hitting_timings) == len(self._ref_hitting_timings) and
            self.double_hit_amp_reward() > 0.3 and
            self.onset_shape_reward() > -15 and
            self.hitting_timing_reward() > 0.7     # this means the timing error is smaller than 0.2 seconds
        ) else 0

def step_amp_reward(prev_setp_rec_audio, step_rec_audio, onset_threshold, ref_audio):
    # TODO: New Step Amp Reward
    if np.mean(abs(step_rec_audio)) - np.mean(abs(prev_setp_rec_audio)) > onset_threshold:
        pass
    pass