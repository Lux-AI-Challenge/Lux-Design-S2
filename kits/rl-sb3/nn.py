import torch as th
import torch.nn as nn
import sys
import zipfile

class Net(nn.Module):
    def __init__(self, action_dims=12):
        super(Net, self).__init__()
        self.action_dims=action_dims
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
        action_logits[~action_masks] = -1e+8 # mask out invalid actions
        distribution = [th.distributions.Categorical(logits=split) for split in th.split(action_logits, tuple([self.action_dims]), dim=1)]
        if not deterministic:
            return th.stack([dist.sample() for dist in distribution], dim=1)
        else:
            return th.stack([dist.mode for dist in distribution], dim=1)

    def forward(self, x):
        x = self.mlp(x)
        return x
import os.path as osp
import io
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


# pol=load_policy("./kits/rl-sb3/best_model.pth")
