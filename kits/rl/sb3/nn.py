"""
Code for neural network inference and loading SB3 model weights
"""
import sys
import zipfile

import torch as th
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, action_dims=12):
        super(Net, self).__init__()
        self.action_dims = action_dims
        self.mlp = nn.Sequential(
            nn.Linear(13, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.action_net = nn.Sequential(
            nn.Linear(128, action_dims),
        )

    def act(self, x, action_masks, deterministic=False):
        latent_pi = self.forward(x)
        action_logits = self.action_net(latent_pi)
        action_logits[~action_masks] = -1e8  # mask out invalid actions
        dist = th.distributions.Categorical(logits=action_logits)
        if not deterministic:
            return dist.sample()
        else:
            return dist.mode

    def forward(self, x):
        x = self.mlp(x)
        return x


import io
import os.path as osp


def load_policy(model_path):
    # load .pth or .zip
    if model_path[-4:] == ".zip":
        with zipfile.ZipFile(model_path) as archive:
            file_path = "policy.pth"
            with archive.open(file_path, mode="r") as param_file:
                file_content = io.BytesIO()
                file_content.write(param_file.read())
                file_content.seek(0)
                sb3_state_dict = th.load(file_content, map_location="cpu")
    else:
        sb3_state_dict = th.load(model_path, map_location="cpu")

    model = Net()
    loaded_state_dict = {}

    # this code here works assuming the first keys in the sb3 state dict are aligned with the ones you define above in Net
    for sb3_key, model_key in zip(sb3_state_dict.keys(), model.state_dict().keys()):
        loaded_state_dict[model_key] = sb3_state_dict[sb3_key]
        print("loaded", sb3_key, "->", model_key, file=sys.stderr)

    model.load_state_dict(loaded_state_dict)
    return model
