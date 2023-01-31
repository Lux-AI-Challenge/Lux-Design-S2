import torch as th
import torch.nn as nn
import sys
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(13, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 12),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.mlp(x)
        x = x.argmax(1)
        return x

def load_policy(model_path):
    sb3_state_dict = th.load(model_path)
    model = Net()
    loaded_state_dict = {}

    # this code here works assuming the first keys in the sb3 state dict are aligned with the ones you define above in Net
    for sb3_key, model_key in zip(sb3_state_dict.keys(), model.state_dict().keys()):
        loaded_state_dict[model_key] = sb3_state_dict[sb3_key]
        print("loaded", sb3_key, "->", model_key, file=sys.stderr)
    
    model.load_state_dict(loaded_state_dict)
    return model


# pol=load_policy("./kits/rl-sb3/best_model.pth")
# import ipdb;ipdb.set_trace()