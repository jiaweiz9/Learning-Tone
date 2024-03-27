"""utils for saving and loading models
"""
import os
import torch

def save_model(model, save_dir, save_name, iteration):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name + "_{}".format(iteration))
    torch.save(model.state_dict(), save_path)


def load_model(model, load_dir, load_name, iteration):
    load_path = os.path.join(load_dir, load_name + "_{}".format(iteration))
    model.load_state_dict(torch.load(load_path))
    return model
