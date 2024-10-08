# [Humanoid 2024] Learning Tone: Towards Robotic Xylophone Mastery

Jiawei Zhang, Taemoon Jeong, Sankalp Yamsani, Sungjoon Choi, and Joohyung Kim

## Installation

```bash
# Install dev-toolkits for portaudio
sudo apt install portaudio19-dev

# Install dependencies
conda env create -n learning-tone python=3.8
conda activate learning-tone
pip install -e .

# Run training
python psyonic_playing_xylopone/train.py
```

## Results
### Thumb Control
<p align="center">
  <img src="docs/images/thumb_low.png" alt="_thumb_low" width="30%">
  <img src="docs/images/thumb_high_new.png" alt="_thumb_high" width="30%">
</p>

- low amplitude:
    - reference: `ref_audio/xylophone_keyB/amp043_05.wav`
    - generated: `results/audios/thumb_low.wav`

- high amplitude:
    - reference: `ref_audio/xylophone_keyB/amp065_015.wav`
    - generated: `results/audios/thumb_high.wav`



### Thumb-Wrist Control
<p align="center">
  <img src="docs/images/wrist_double.png" alt="wrist_double" width="30%">
    <img src="docs/images/wrist_low.png" alt="wrist_low" width="30%">
    <img src="docs/images/wrist_high.png" alt="wrist_high" width="30%">
</p>

- double hit:
    - reference: `ref_audio/xylophone_keyB/double06.wav`
    - generated: `results/audios/wrist_double.wav`

- low amplitude:
    - reference: `ref_audio/xylophone_keyB/amp045_025.wav`
    - generated: `results/audios/wrist_low.wav`

- high amplitude:
    - reference: `ref_audio/xylophone_keyB/amp06_013.wav`
    - generated: `results/audios/wrist_high.wav`


