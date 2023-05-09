from numpy.lib.arraysetops import isin
import torch
import torch.nn.functional as F
import os
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import SequentialNetwork, PolicyNetworkFactory, get_optimizer
import numpy as np
from unstable_baselines.common import util, functional
from operator import itemgetter

class MBPOAgent(BaseAgent):
    def __init__(self,observation_space, action_space, env_name,
        target_smoothing_tau: float,
        alpha: float,
        reward_scale: float,
        **kwargs):
        super(MBPOAgent, self).__init__()
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        #save parameters
        self.args = kwargs
        
        #initilze networks
        self.q1_network = SequentialNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.q2_network = SequentialNetwork(obs_dim + action_dim, 1,**kwargs['q_network'])
        self.target_q1_network = SequentialNetwork(obs_dim + action_dim, 1,**kwargs['q_network'])
        self.target_q2_network = SequentialNetwork(obs_dim + action_dim, 1,**kwargs['q_network'])
        self.policy_network = PolicyNetworkFactory.get(observation_space, action_space,  ** kwargs['policy_network'])
        
        #sync network parameters
        functional.soft_update_network(self.q1_network, self.target_q1_network, 1.0)
        functional.soft_update_network(self.q2_network, self.target_q2_network, 1.0)

        #pass to util.device
        self.q1_network = self.q1_network.to(util.device)
        self.q2_network = self.q2_network.to(util.device)
        self.target_q1_network = self.target_q1_network.to(util.device)
        self.target_q2_network = self.target_q2_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)
        
        #initialize optimizer
        self.q1_optimizer = get_optimizer(network = self.q1_network, **kwargs['q_network'])
        self.q2_optimizer = get_optimizer(network = self.q2_network, **kwargs['q_network'])
        self.policy_optimizer = get_optimizer(network = self.policy_network, **kwargs['policy_network'])

        #hyper-parameters
        self.gamma = kwargs['gamma']
        self.automatic_entropy_tuning = kwargs['entropy']['automatic_tuning']
        target_entropy = kwargs['entropy']['target_entropy']
        self.alpha = alpha
        if self.automatic_entropy_tuning is True:
            if target_entropy == 'auto':
                self.target_entropy = -np.prod(action_space.shape).item()
            else:
                assert isinstance(target_entropy, int) or isinstance(target_entropy, float)
                self.target_entropy = target_entropy

            self.log_alpha = torch.zeros(1, requires_grad=True, device=util.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=kwargs['entropy']['learning_rate'])
        self.tot_update_count = 0 
        self.target_smoothing_tau = target_smoothing_tau
        self.reward_scale = reward_scale


    def update(self, data_batch):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, truncated_batch = \
            itemgetter("obs", "action", "reward", "next_obs", "done", "truncated")(data_batch)
        
       
        reward_batch = reward_batch * self.reward_scale
        
        #calculate critic loss
        curr_q1 = self.q1_network([obs_batch, action_batch])
        curr_q2 = self.q2_network([obs_batch, action_batch])
        with torch.no_grad():
            next_obs_action, next_obs_log_prob = \
                itemgetter("action", "log_prob")(self.policy_network.sample(next_obs_batch))

            target_q1_value = self.target_q1_network([next_obs_batch, next_obs_action])
            target_q2_value = self.target_q2_network([next_obs_batch, next_obs_action])
            next_obs_min_q = torch.min(target_q1_value, target_q2_value)
            target_q = (next_obs_min_q - self.alpha * next_obs_log_prob)
            target_q = reward_batch + self.gamma * (1. - done_batch) * target_q

        #compute q loss and backward
        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        ##########

        new_curr_obs_action, new_curr_obs_log_prob = \
            itemgetter("action", "log_prob")(self.policy_network.sample(obs_batch))
        new_curr_obs_q1_value = self.q1_network([obs_batch, new_curr_obs_action])
        new_curr_obs_q2_value = self.q2_network([obs_batch, new_curr_obs_action])
        new_min_curr_obs_q_value = torch.min(new_curr_obs_q1_value, new_curr_obs_q2_value)
        #print(new_curr_obs_log_prob.shape, self.target_entropy, new_curr_obs_log_prob.mean())
        #compute policy and ent loss
        policy_loss = ((self.alpha * new_curr_obs_log_prob) - new_min_curr_obs_q_value).mean()
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_curr_obs_log_prob + self.target_entropy).detach()).mean()
            alpha_loss_value = alpha_loss.detach().cpu().item()
            self.alpha_optim.zero_grad()
        else:
            alpha_loss = 0.
            alpha_loss_value = 0.
        
        self.policy_optimizer.zero_grad()
        (policy_loss + alpha_loss).backward()
        self.policy_optimizer.step()
        if self.automatic_entropy_tuning:
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()

        # backward and step
        self.tot_update_count += 1

        self.update_target_network()
        
        return {
            "loss/q1": q1_loss.item(), 
            "loss/q2": q2_loss.item(), 
            "loss/policy": policy_loss.item(), 
            "loss/entropy": alpha_loss_value, 
            "misc/entropy_alpha": self.alpha.item(), 
            "misc/train_reward_mean": torch.mean(reward_batch).item(),
            "misc/train_reward_var": torch.var(reward_batch).item()
        }
        

    def update_target_network(self):
        functional.soft_update_network(self.q1_network, self.target_q1_network, self.target_smoothing_tau)
        functional.soft_update_network(self.q2_network, self.target_q2_network, self.target_smoothing_tau)

    @torch.no_grad()  
    def select_action(self, obs, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None,]
    
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(util.device)

        action_scaled, log_prob = \
                itemgetter("action", "log_prob")(self.policy_network.sample(obs, deterministic))
        return {
            "action": action_scaled.detach().cpu().numpy(), 
            "log_prob": log_prob.detach().cpu().numpy()[0]
        }


    

        



