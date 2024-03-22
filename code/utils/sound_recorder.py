import numpy as np
import sounddevice as sd
import queue
import threading

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

    def start_recording(self):
        if not self.is_recording:
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
                self.q.put(audio_chunk)
                # print("put audio chunk")
            print("Recording stopped!")

    def stop_recording(self):
        # Stops the ongoing audio recording.
        self.is_recording = False
        if self.recording_thread is not None:
            self.recording_thread.join()
            self.recording_thread = None

    def clear_buffer(self):
        assert self.is_recording == False
        self.recording_list = []
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
            return []