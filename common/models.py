
#implement model to learn state transitions and rewards
import torch
from unstable_baselines.common.networks import get_network, get_act_cls, EnsembleMLPNetwork
import torch.nn as nn
from abc import abstractmethod
import numpy as np
import torch.nn.functional as F
from unstable_baselines.common import util 


class BaseModel(nn.Module):
    def __init__(self,obs_dim, action_dim, hidden_dims, out_act_fn="identity", **kwargs):
        super(BaseModel, self).__init__()
        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        hidden_dims = [obs_dim + action_dim] + hidden_dims 
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.networks = []
        out_act_cls = get_act_cls(out_act_fn)
        for i in range(len(hidden_dims)-1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i+1]
            curr_network = get_network([curr_shape, next_shape])
            self.networks.extend([curr_network, Swish()])
        self.output_dim = obs_dim + 1
        final_network = get_network([hidden_dims[-1], self.output_dim * 2])
        self.networks.extend([final_network, out_act_cls()])
        self.networks = nn.Sequential(*self.networks)

        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10), requires_grad=False)

    def forward(self, state, action):
        ip = torch.cat([state, action], 1)
        out = self.networks(ip)

        mean = out[:, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - out[:, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, torch.exp(logvar)
    
    def predict(self, state, action):
        pass



class EnsembleModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims, device, ensemble_size = 7, num_elite=5, weight_decays=None ,act_fn="swish", out_act_fn="identity", reward_dim=1,**kwargs):
        super(EnsembleModel, self).__init__()
        assert(weight_decays is None or len(weight_decays) == len(hidden_dims) + 1)
        self.out_dim = obs_dim + reward_dim
        self.model = EnsembleMLPNetwork(input_dim=obs_dim+action_dim, out_dim=self.out_dim*2, ensemble_size=ensemble_size, hidden_dims=hidden_dims, act_fn=act_fn, out_act_fn=out_act_fn, **kwargs)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_elite = num_elite
        self.elite_model_idxes = torch.tensor([i for i in range(num_elite)])
        self.max_logvar = nn.Parameter((torch.ones((1, self.out_dim)).float() / 2).to(device), requires_grad=True)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.out_dim)).float() * 10).to(device), requires_grad=True)
        self.add_module("model", self.model)
        self.to(device)

    def predict(self, obs, action):
        # model_input = torch.cat([state, action], 1)
        if type(obs) != torch.Tensor:
            if len(obs.shape) == 1:
                obs = torch.FloatTensor([obs]).to(util.device)
                action = torch.FloatTensor([action]).to(util.device)
            else:
                obs = torch.FloatTensor(obs).to(util.device)
                action = torch.FloatTensor(action).to(util.device)
        predictions = self.model(torch.cat([obs, action], axis=1))

        mean = predictions[:, :, :self.out_dim]
        logvar = predictions[:, :, self.out_dim:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        
        return mean, logvar
    
    def get_decay_loss(self):
        return self.model.get_decay_loss()


if __name__ == "__main__":
    device = torch.device("cpu")
    model = EnsembleModel(10, 3, [20, 20], device)
    for p in model.parameters():
        print(p)
