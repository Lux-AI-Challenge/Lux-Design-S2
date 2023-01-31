import torch as th
import torch.nn as nn
import sys
class Net(nn.Module):
    def __init__(self, action_dims=12):
        super(Net, self).__init__()
        self.action_dims=action_dims
        self.mlp = nn.Sequential(
            nn.Linear(13, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dims),
            nn.Tanh()
        )

    def act(self, x, deterministic=False):
        action_logits = self.forward(x)
        if not deterministic:
            distribution = [th.distributions.Categorical(logits=split) for split in th.split(action_logits, tuple([self.action_dims]), dim=1)]
            return th.stack([dist.sample() for dist in distribution], dim=1)
        else:
            return x.argmax(1)

    def forward(self, x):
        x = self.mlp(x)
        return x

def load_policy(model_path):
    sb3_state_dict = th.load(model_path)
    model = Net()
    loaded_state_dict = {}
    # import ipdb;ipdb.set_trace()
    # this code here works assuming the first keys in the sb3 state dict are aligned with the ones you define above in Net
    for sb3_key, model_key in zip(sb3_state_dict.keys(), model.state_dict().keys()):
        loaded_state_dict[model_key] = sb3_state_dict[sb3_key]
        print("loaded", sb3_key, "->", model_key, file=sys.stderr)
    
    model.load_state_dict(loaded_state_dict)
    return model


# pol=load_policy("./kits/rl-sb3/best_model.pth")
