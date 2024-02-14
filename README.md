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

### Usage
Explain how to use your project. Provide examples.

### Running the tests
Explain how to run the automated tests for this system.

### Deployment
Add additional notes about how to deploy this on a live system.

### Contributing
Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

### License
This project is licensed under the MIT License - see the LICENSE.md file for details

### Acknowledgments
Hat tip to anyone whose code was used.
Inspiration.
etc.
### Contact
Taemoon Jeong, taemoon-jeong@korea.ac.kr
