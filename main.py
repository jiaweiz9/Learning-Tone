import numpy as np
import librosa
import matplotlib.pyplot as plt
import wavio
import os

def __visualize_audio(ref_audio, rec_audio, rec_idx, step_rews, sr=44100) -> None:
    
    plt.figure(figsize=(20,6))
    time = np.arange(0, len(rec_audio)) / 44100
    plt.plot(time, rec_audio, color='blue', alpha=0.3)

    ref_audio = np.pad(ref_audio, (0, len(rec_audio) - len(ref_audio)), 'constant')
    plt.plot(time, ref_audio, color='red', alpha=0.3)

    rec_idx = rec_idx * 0.02
    plt.scatter(rec_idx, step_rews, color='black', alpha=0.5, marker='x')

    # plt.title(f'Episode {rec_idx} - Reference Audio (red) vs. Recorded Audio (blue)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(['Recorded Audio', 'Reference Audio', 'Step Reward'])

    # file_name = f"episode_{self.num_timesteps}.png"
    # img_path = os.path.join(self.figures_path, file_name)
    plt.savefig("1.png")
    plt.close()

def __ref_chunks_mean_nearby(rec_chunk_idx, before, after):
    # find the reference audio chunks in the range [rec_chunk_idx - before, rec_chunk_idx + after]
    if np.mean(abs(rec_audio[rec_chunk_idx*882: (rec_chunk_idx+1)*882])) < 0.02:
        return ref_audio[rec_chunk_idx*882: (rec_chunk_idx+1)*882], 0
    else:
        start_idx = max(rec_chunk_idx - before, 0)
        end_idx = min(rec_chunk_idx + after, len(ref_audio) // 882 - 1)
        min_diff = 10
        for i in range(start_idx, end_idx):
            ref_chunk = ref_audio[i*882: (i+1)*882]
            rec_chunk = rec_audio[rec_chunk_idx*882: (rec_chunk_idx+1)*882]
            diff = abs(np.mean(abs(ref_chunk)) -np.mean(abs(rec_chunk)))
            if diff < min_diff:
                min_diff = diff
                min_diff_idx = i
        return ref_audio[min_diff_idx*882: (min_diff_idx+1)*882], min_diff
        


ref_audio = librosa.load("ref_audio/xylophone/ref_hit1_clip.wav", sr=44100)[0]
rec_audio = librosa.load("ref_audio/xylophone/ref_hit1_clip.wav", sr=44100)[0]
rec_audio = np.pad(rec_audio, (1000, 0), 'constant')

rec_index = np.arange(0, 100)

step_rews = []
for i in range(100):
    rec_audio_step = rec_audio[i*882: (i+1)*882]
    ref_audio_step, mean_diff = __ref_chunks_mean_nearby(rec_index[i], 5, 5)
    step_rews.append(-abs(np.mean(abs(ref_audio_step)) -np.mean(abs(rec_audio_step))))
    # step_rews.append(-mean_diff)

step_rews = np.array(step_rews)

__visualize_audio(ref_audio, rec_audio, rec_index, step_rews, sr=44100)

# list_reward = []
# reward = np.array(1)
# print(reward)

# list_reward.append(reward.copy())
# print(list_reward)
# reward += 1
# print(list_reward)