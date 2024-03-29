# Making-sound-with-hand

Generating Realistic Sound with Prosthetic Hand: A Reinforcement Learning Approach [EMBC 2024 Submitted]

## Getting Started
### Prerequisites
Ensure `pyenv` and `pyenv-virtualenv` are installed on your system.
If not, you can find installation instructions on the [`pyenv` GitHub page](https://github.com/pyenv/pyenv#installation) and the [`pyenv-virtualenv` GitHub page](https://github.com/pyenv/pyenv-virtualenv).

### Installing Python 3.8.10 with pyenv
First, install Python 3.8.10 using `pyenv`:
```bash
pyenv install 3.8.10
```

### Creating a Virtual Environment
With `pyenv-virtualenv`, you can create a virtual environment for the installed Python version:
```bash
pyenv virtualenv 3.8.10 myproject-env
```
Replace `myproject-env` with a name that's relevant to your project. This command creates a virtual environment named `myproject-env` using Python 3.8.10.

### Activating the Virtual Environment
To activate the virtual environment for your project, navigate to your project directory and set the local Python version to your virtual environment:
```bash
pyenv local myproject-env
```
This command tells `pyenv` to use `myproject-env` as the active Python version in your project directory.

### Installing Dependencies
With the virtual environment activated, install your project's dependencies:
```bash
pip install -r requirements.txt
```
This installs all required packages as specified in `requirements.txt` file.

## Usage
Our system utilizes the following hardware components:
- PSYONIC Ability Hand
- PAPRAS robot arm
- ZOOM H6 Recorder
- Eastar Drum Practice Pad (8-inch)

Make sure to have these components set up and connected appropriately before running the software.

### Running the tests
```bash
python main_psyonic_sound_v7.py --seed 111 --record_duration 4 --samplerate 44100 --beta_dist --max_iter 10
```


## Contact
Taemoon Jeong, taemoon-jeong@korea.ac.kr


## Denoise
Download NoiseTorch: [text](https://github.com/noisetorch/NoiseTorch)

## Init Position setup
rostopic pub robot1/psyonic_controller std_msgs/Float32MultiArray "layout:
  dim:
  - label: ''
    size: 0
    stride: 0
  data_offset: 0
data:
- 105
- 105
- 105
- 110
- 70
- -0
"
