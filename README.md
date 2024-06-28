# Psyonic Hand Playing Xylophone

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
python main_psyonic_sound_v7.py --record_duration 2 --samplerate 44100 --max_iter 1000 --n_epi 10 --WANDB --SAVE_WEIGHTS --seed 
```
rosrun rosserial_python serial_node.py /dev/ttyACM0 _baud:=4000000

## Contact
Taemoon Jeong, taemoon-jeong@korea.ac.kr


<!-- ## Denoise
Download NoiseTorch: [text](https://github.com/noisetorch/NoiseTorch) -->

## Initial Position setup
```

// connect pysonic
cd ws_music && source devel/setup.bash && rosrun rosserial_python serial_node.py /dev/ttyACM0 _baud:=4000000

cd ws_music && source devel/setup.bash && roslaunch papras_table_demo hw_joint6.launch

rostopic pub /joint6_controller/command std_msgs/Float64MultiArray "layout:
  dim:
  - label: ''
    size: 0
    stride: 0
  data_offset: 0
data:
- 0
"

wrist position for thumb-control: 0

rostopic pub robot1/psyonic_controller std_msgs/Float32MultiArray "layout:
  dim:
  - label: ''
    size: 0
    stride: 0
  data_offset: 0
data:
- 50
- 70
- 110
- 115
- 50
- -10
"
```



## Results

Experiment: 0613_2243-bgpv1jyk

|        amplitude        |  0.9158610435469906 |
|      hitting_times      |          0          |
|       onset_shape       | 0.02080897135699532 |
|      hitting_timing     |  0.9419501133786848 |
|      success reward     |  76.77989182893056  |
|       Moving back       |          20         |
| Episode moving distance |         110         |

python test.py epi_length=50 ++short_epi=True referen ++load_model_path=