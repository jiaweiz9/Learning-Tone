import numpy as np
import sounddevice as sd
import queue
import threading
import time

class SoundRecorder():
    def __init__(self, samplerate=44100, audio_device=None):
        self.stream = None
        self.is_recording = False
        self.recording_list = []  # Using list instead of np.array for efficiency during recording

        self.q = queue.Queue()
        self.sample_rate = samplerate
        self.recording_thread = None
        self.lock = threading.Lock()
        self.audio_idx = audio_device
        self.audio_buffer = np.array([])

    def start_recording(self):
        if not self.is_recording:
            print("enter recording")
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            print("Recording started!")
        else:
            print("Already recording!")

    def _record_audio(self):
        chunk_size = 882  # 20ms
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', device=self.audio_idx) as stream:
            print('Starting recording...')
            while self.is_recording:
                audio_chunk, overflowed = stream.read(chunk_size)
                if overflowed:
                    print("Audio buffer overflowed! Some audio might be lost.")
                
                # self.recording_list.append(audio_chunk)
                # self.q.put(audio_chunk)
                self.audio_buffer = np.append(self.audio_buffer, audio_chunk)
        print("Recording stopped!")

    def get_last_step_audio(self):
        '''
        Returns the last 20ms audio chunk and the index of the chunk in the episode.
        '''
        return self.audio_buffer[-882:], len(self.audio_buffer)//882 - 1
    
    def get_episode_audio(self):
        return self.audio_buffer[:]

    def stop_recording(self):
        # Stops the ongoing audio recording.
        self.is_recording = False
        if self.recording_thread is not None:
            self.recording_thread.join()
            self.recording_thread = None

    def clear_buffer(self):
        assert self.is_recording == False
        self.recording_list = []
        self.audio_buffer = np.array([])
        while not self.q.empty():
            self.q.get()
        assert self.q.empty() == True

    def get_current_buffer(self):

        # if self.recording_list:
        #     # Convert list of chunks to a single numpy array
        #     return np.concatenate(self.recording_list, axis=0)
        # else:
        #     print("Recording hasn't started yet!")
        #     return None
        while True:
            try:
                self.recording_list.append(self.q.get_nowait())
            except queue.Empty:
                break
        # print("length of recording list", len(self.recording_list))
        if self.recording_list:
            return np.concatenate(self.recording_list, axis=0)
        else:
            print("Recording hasn't started yet!")
            return np.array([])
        
    def record_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.q.put(indata[:, 0])

    def setup_stream(self):
        self.stream = sd.InputStream(samplerate = self.sample_rate, channels = 1, dtype='float32', 
                                     device = self.audio_idx, callback = self.record_callback)
        



class OldSoundRecorder():
    def __init__(self, samplerate=44100, audio_device=None):
        self.stream = None
        self.is_recording = False
        self.recording_list = []  # Using list instead of np.array for efficiency during recording
        self.sample_rate = samplerate
        self.recording_thread = None
        self.lock = threading.Lock()
        self.audio_idx = audio_device

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
        else:
            print("Already recording!")

    def _record_audio(self):
        chunk_size = 441  # 20ms
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', device=self.audio_idx) as stream:
            print('Starting recording...')
            while self.is_recording:
                audio_chunk, overflowed = stream.read(chunk_size)
                if overflowed:
                    print("Audio buffer overflowed! Some audio might be lost.")
                with self.lock:
                    self.recording_list.append(audio_chunk)
            print("Recording stopped!")

    def stop_recording(self):
        # Stops the ongoing audio recording.
        self.is_recording = False
        if self.recording_thread is not None:
            self.recording_thread.join()
            self.recording_thread = None

    def get_current_buffer(self):
        with self.lock:
            if self.recording_list:
                # Convert list of chunks to a single numpy array
                return np.concatenate(self.recording_list, axis=0)
            else:
                print("Recording hasn't started yet!")
                return None


if __name__ == "__main__":
    sound_recorder = SoundRecorder()
    sound_recorder.start_recording()
    time.sleep(0.1)
    # time.sleep(5)
    # data = sound_recorder.get_current_buffer()
    # print(len(data)//882)
    index_list = []

    for i in range(100):
        time.sleep(0.025)
        data, index = sound_recorder.get_last_step_audio()
        print(len(data), " ", index)
        index_list.append(index - 5)
    sound_recorder.stop_recording()

    data = sound_recorder.get_episode_audio()
    print(len(data[4410:])//882)
    print(index_list)
    sound_recorder.clear_buffer()