import numpy as np
import torch
import os
from unstable_baselines.common import util
from unstable_baselines.common.networks import get_optimizer
from unstable_baselines.model_based_rl.common.models import EnsembleModel
from operator import itemgetter
from unstable_baselines.common.scaler import StandardScaler
class TransitionModel:
    def __init__(self, 
            obs_space, 
            action_space, 
            env_name, 
            model_batch_size=256,
            holdout_ratio=0.1, 
            inc_var_loss=False, 
            use_weight_decay=False,
            **kwargs):
        
        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]
        self.model = EnsembleModel(obs_dim=obs_dim, action_dim=action_dim, device=util.device, **kwargs['model'])
        self.env_name = env_name
        self.model_optimizer = get_optimizer(optimizer_class=kwargs['optimizer_class'], network=self.model, learning_rate=kwargs['learning_rate'] )
        self.networks = {
            "model": self.model
        }
        self.holdout_ratio = holdout_ratio
        self.inc_var_loss = inc_var_loss
        self.use_weight_decay = use_weight_decay
        self.obs_scaler = StandardScaler()
        self.act_scaler = StandardScaler()


    def _termination_fn(self, env_name, obs, act, next_obs):
        # TODO: add more done function
        if env_name == "Hopper-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
            return done
        elif env_name == "Walker2d-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            return done
        elif 'walker_' in env_name:
            torso_height =  next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in env_name or 'walker_5' in env_name:
                offset = 0.
            else:
                offset = 0.26
            not_done = (torso_height > 0.8 - offset) \
                       * (torso_height < 2.0 - offset) \
                       * (torso_ang > -1.0) \
                       * (torso_ang < 1.0)
            done = ~not_done
            return done
        elif "Swimmer" in env_name or "HalfCheetah" in env_name: # No done for these two envs
            return np.array([False for _ in obs])
        else:
            raise NotImplementedError

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    @torch.no_grad()
    def select_elite_models(self, data):
        obs_list, action_list, next_obs_list, reward_list = \
            itemgetter("obs",'action','next_obs', 'reward')(data)
        
        model_input = torch.cat([obs_list, action_list], dim=-1)

        predictions = self.model.predict(model_input)
        groundtruths = torch.cat((next_obs_list - obs_list, reward_list), dim=1)
        test_mean_losses, test_var_losses  = self.model_loss(predictions, groundtruths)
        test_losses = (test_mean_losses + test_var_losses).detach().cpu().numpy()
        idx = np.argsort(test_losses)
        self.model.elite_model_idxes = idx[:self.model.num_elite]
        
    def update_scaler(self, obs, action):
        self.obs_scaler.update(obs)
        self.act_scaler.update(action)
    
    def transform_obs_action(self, obs, action):
        obs = self.obs_scaler.transform(obs)
        action = self.act_scaler.transform(action)
        return obs, action

    def update(self, data_batch):
        #compute the number of holdout samples
        batch_size = data_batch['obs'].shape[0]
        num_holdout = int(batch_size * self.holdout_ratio)

        #permutate samples
        obs_batch, action_batch, next_obs_batch, reward_batch = \
            itemgetter("obs",'action','next_obs', 'reward')(data_batch)

        delta_obs_batch = next_obs_batch - obs_batch
        self.update_scaler(obs_batch, action_batch)
        obs_batch, action_batch = self.transform_obs_action(obs_batch, action_batch)

        #predict with model
        model_input = torch.cat([obs_batch, action_batch], dim=-1)
        predictions = self.model.predict(model_input)
        
        #compute training loss
        groundtruths = torch.cat((delta_obs_batch, reward_batch), dim=1)
        train_mean_losses, train_var_losses = self.model_loss(predictions, groundtruths)
        train_mean_loss = torch.sum(train_mean_losses)
        train_var_loss = torch.sum(train_var_losses)
        train_transition_loss = train_mean_loss + train_var_loss
        train_transition_loss += 0.01 * torch.sum(self.model.max_logvar) - 0.01 * torch.sum(self.model.min_logvar) # why
        if self.use_weight_decay:
            decay_loss = self.model.get_decay_loss()
            train_transition_loss += decay_loss
        else:
            decay_loss = None
        #udpate transition model
        self.model_optimizer.zero_grad()
        train_transition_loss.backward()
        self.model_optimizer.step()

        #compute testing loss for elite model
        
        return {
            "loss/train_transition_loss_mean": train_mean_loss.item(),
            "loss/train_transition_loss_var": train_var_loss.item(),
            "loss/train_transition_loss": train_var_loss.item(),
            "loss/decay_loss": decay_loss.item() if decay_loss is not None else 0,
            "misc/max_logvar_mean": self.model.max_logvar.mean().item(),
            "misc/max_logvar_var": self.model.max_logvar.var().item(),
            "misc/min_logvar_mean": self.model.min_logvar.mean().item(),
            "misc/max_logvar_var": self.model.min_logvar.var().item()
        }

    def model_loss(self, predictions, groundtruths):
        pred_means, pred_logvars = predictions
        if self.inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            inv_var = torch.exp(-pred_logvars)
            mean_losses = torch.mean(torch.mean(torch.pow(pred_means - groundtruths, 2) * inv_var, dim=-1), dim=-1)
            var_losses = torch.mean(torch.mean(pred_logvars, dim=-1), dim=-1)
        else:
            mean_losses = torch.mean(torch.pow(pred_means - groundtruths, 2), dim=(1, 2))
            var_losses = None

        return mean_losses, var_losses

    @torch.no_grad()  
    def predict(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None,]
            act = act[None,]
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(util.device)
        if not isinstance(act, torch.Tensor):
            act = torch.FloatTensor(act).to(util.device)

        scaled_obs, scaled_act = self.transform_obs_action(obs, act)
        
        model_input = torch.cat([scaled_obs, scaled_act], dim=-1)
        pred_means, pred_logvars = self.model.predict(model_input)
        pred_means = pred_means.detach().cpu().numpy()
        pred_vars = pred_logvars.exp().detach().cpu().numpy()
        #add curr obs for next obs
        obs = obs.detach().cpu().numpy()
        act = act.detach().cpu().numpy()
        pred_means[:, :, :-1] += obs[None,].repeat(pred_means.shape[0], axis=0)
        ensemble_model_stds = pred_logvars.detach().cpu().numpy()

        if deterministic:
            ensemble_samples = pred_means
        else:
            ensemble_samples = pred_means + np.random.normal(size=pred_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = pred_means.shape
        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = pred_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        log_prob, dev = self._get_logprob(samples, pred_means, pred_vars)

        next_obs, rewards = samples[:, :-1] + obs, samples[:, -1]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:, :-1], terminals[:,None], model_means[:, -1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:, :-1], np.zeros((batch_size, 1)), model_stds[:, -1:]), axis=-1)

        assert(type(next_obs) == np.ndarray)

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info

    # def forward(self, obs, act, deterministic=False):
    #     pred_means, pred_logvars = self.model.predict(obs, act)
    #     ensemble_model_means, ensemble_model_vars = [], []
    #     for (mean, var) in zip(torch.unbind(pred_means), torch.unbind(pred_logvars)):
    #         ensemble_model_means.append(mean)
    #         ensemble_model_vars.append(var.exp())
    #     ensemble_model_means, ensemble_model_vars = \
    #         torch.stack(ensemble_model_means), torch.stack(ensemble_model_vars)
    #     ensemble_model_means[:, :, :-1] += obs
    #     ensemble_model_stds = torch.sqrt(ensemble_model_vars)

    #     if deterministic:
    #         ensemble_samples = ensemble_model_means
    #     else:
    #         ensemble_samples = ensemble_model_means + torch.randn_like(ensemble_model_means.shape) * ensemble_model_stds

    #     num_models, batch_size, _ = ensemble_model_means.shape

    #     model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
    #     batch_idxes = np.arange(0, batch_size)

    #     samples = ensemble_samples[model_idxes, batch_idxes]

    #     next_obs, rewards = samples[:, :-1] + obs, samples[:, -1]

    #     return next_obs, rewards

    def save_model(self, epoch):
        save_dir = os.path.join(util.logger.log_path, 'models')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_dir = os.path.join(save_dir, "ite_{}".format(epoch))
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        for network_name, network in self.networks.items():
            save_path = os.path.join(model_save_dir, network_name + ".pt")
            torch.save(network, save_path)