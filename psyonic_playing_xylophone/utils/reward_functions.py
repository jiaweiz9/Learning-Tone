import numpy as np
from numpy.typing import NDArray
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from typing import List, Any, Dict, Literal

class RecRefRewardFunction:
    def __init__(self, rec_audio:NDArray[Any]=None, 
                 ref_audio: NDArray[Any]=None, 
                 episode_length: int=100,
                 sr: int=44100):
        self.rec_audio = rec_audio
        self.ref_audio = ref_audio
        self.episode_length = episode_length
        self.sr = sr

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
        hit_amp_th = 0.02
        self.rec_max_amp = np.max(abs(self.rec_audio))
        if self.rec_max_amp < hit_amp_th:
            self.rec_audio[:] = 1e-5
        self.ref_max_amp = np.max(abs(self.ref_audio))
        print("ref_audio shape: ", self.ref_audio.shape)
        print("rec_audio shape: ", self.rec_audio.shape)


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

        # print(f"hitting frames reference {self._ref_hitting_frames}")
        # print(f"hitting frames recorded {self._rec_hitting_frames}")


    def amplitude_reward(self, amp_scale=1e2):
        max_amp_diff = np.abs(self.rec_max_amp - self.ref_max_amp)
        return np.exp(-max_amp_diff * 10)


    def hitting_times_reward(self) -> float:
        # # if no hit, return -10
        # if len(self._rec_hitting_timings) == 0:
        #     return -10
        # # the returned difference will not be smaller than -20
        # else:
        #     return -min(abs(len(self._rec_hitting_timings) - len(self._ref_hitting_timings)), 20)
        return -min(abs(len(self._rec_hitting_timings) - len(self._ref_hitting_timings)), 20)
        

    # Compute the DTW distance (Dynamic Time Warping) between the onset strength envelops of the recorded and reference audio, serving as a measure of shape similarity
    def onset_shape_reward(self) -> float:
        rec_audio_8k = librosa.resample(self.rec_audio, orig_sr=self.sr, target_sr=8000)
        ref_audio_8k = librosa.resample(self.ref_audio, orig_sr=self.sr, target_sr=8000)
        dtw_difference, _ = fastdtw(rec_audio_8k, ref_audio_8k)
        # print(dtw_difference)
        return np.exp(-dtw_difference / 50)


    def hitting_timing_reward(self) -> float:
        timing_partition_rec = self._rec_hitting_frames / len(self.rec_audio)
        timing_partition_ref = self._ref_hitting_frames / len(self.ref_audio)
        # if no hit, return 0
        print(f"ref hitting: {timing_partition_ref}")
        print(f"rec hitting: {timing_partition_rec}")

        if len(self._rec_hitting_frames) == len(self._ref_hitting_frames):
            # time_diff = abs(np.sum(onset_hit_times_ref - onset_hit_times_rec))
            timing_diff = abs(np.sum(timing_partition_rec - timing_partition_ref))
            return 1 - 4 * timing_diff ** 2 if timing_diff < 0.5 else 0
        return 0
