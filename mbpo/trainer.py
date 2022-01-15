from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
import cv2
import os
from tqdm import  trange
import torch
import random
from unstable_baselines.common import util, functional
from unstable_baselines.common.functional import dict_batch_generator

class MBPOTrainer(BaseTrainer):
    def __init__(self, agent, train_env, eval_env, transition_model, env_buffer, model_buffer, rollout_step_generator,
            agent_batch_size=100,
            model_batch_size=256,
            rollout_batch_size=100000,
            rollout_mini_batch_size=1000,
            model_retain_epochs=1,
            num_env_steps_per_epoch=1000,
            train_model_interval=250,
            train_agent_interval=1,
            num_agent_updates_per_env_step=20, # G
            max_epoch=100000,
            max_agent_updates_per_env_step=5,
            max_model_update_epochs_to_improve=5,
            max_model_train_iterations="None",
            warmup_timesteps=5000,
            model_env_ratio=0.8,
            hold_out_ratio=0.1,
            load_dir="",
            **kwargs):
        super(MBPOTrainer, self).__init__(agent, train_env, eval_env, **kwargs)
        self.agent = agent
        self.env_buffer = env_buffer
        self.model_buffer = model_buffer
        self.train_env = train_env 
        self.eval_env = eval_env
        self.transition_model = transition_model
        self.rollout_step_generator = rollout_step_generator 
        #hyperparameters
        self.agent_batch_size = agent_batch_size
        self.model_batch_size = model_batch_size
        self.rollout_batch_size = rollout_batch_size
        self.rollout_mini_batch_size = rollout_mini_batch_size
        self.model_retain_epochs = model_retain_epochs
        self.num_env_steps_per_epoch = num_env_steps_per_epoch
        self.train_agent_interval = train_agent_interval
        self.train_model_interval = train_model_interval
        self.num_agent_updates_per_env_step = num_agent_updates_per_env_step
        self.max_agent_updates_per_env_step = max_agent_updates_per_env_step
        self.max_model_update_epochs_to_improve = max_model_update_epochs_to_improve
        if max_model_train_iterations == "None":
            self.max_model_train_iterations = np.inf
        else:
            self.max_model_train_iterations = max_model_train_iterations
        self.max_epoch = max_epoch
        self.warmup_timesteps = warmup_timesteps
        self.model_env_ratio = model_env_ratio
        self.hold_out_ratio = hold_out_ratio
        if load_dir != "":
            self.agent.load(load_dir)
        self.model_tot_train_timesteps = 0

    def warmup(self):
        obs = self.train_env.reset()
        for step in tqdm(range(self.warmup_timesteps)):
            action = self.train_env.action_space.sample()
            next_obs, reward, done, info = self.train_env.step(action)
            self.env_buffer.add_transition(obs, action, next_obs, reward, float(done))
            obs = next_obs
            if done:
                obs = self.train_env.reset()


    def train(self):
        epoch_durations = []
        train_traj_returns = [0]
        train_traj_lengths = [0]
        traj_return = 0
        traj_length = 0
        tot_agent_update_steps = 0

        log_dict = self.evaluate()
        for log_key in log_dict:
            util.logger.log_var(log_key, log_dict[log_key], 0)
        util.logger.log_str("Warming Up")

        self.warmup()
        tot_env_steps = self.warmup_timesteps
        
        model_rollout_steps = int(self.rollout_step_generator.initial_val)
        self.resize_model_buffer(model_rollout_steps)
        obs = self.train_env.reset()
        done = False
        
        util.logger.log_str("Started Training")

        for epoch in trange(self.max_epoch, colour='blue', desc='outer loop'): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            
            epoch_start_time = time()

            new_model_rollout_steps = int(self.rollout_step_generator.next())
            if epoch == 0 or new_model_rollout_steps != model_rollout_steps:
                self.resize_model_buffer(new_model_rollout_steps)
                util.logger.log_var("model/model_buffer_size", self.model_buffer.max_buffer_size, tot_env_steps)
                util.logger.log_var("model/rollout_step", new_model_rollout_steps, tot_env_steps)

            for env_step in trange(self.num_env_steps_per_epoch, colour='green', desc='inner loop'):

                self.pre_iter()
                log_infos = {}

                action = self.agent.select_action(obs)['action']
                next_obs, reward, done, _ = self.train_env.step(action)
                tot_env_steps += 1
                traj_length  += 1
                traj_return += reward
                self.env_buffer.add_transition(obs, action, next_obs, reward, float(done))
                obs = next_obs
                if done or traj_length >= self.max_trajectory_length:
                    obs = self.train_env.reset()
                    train_traj_returns.append(traj_return)
                    train_traj_lengths.append(traj_length)
                    traj_length = 0
                    traj_return = 0
                log_infos["performance/train_return"] = train_traj_returns[-1]
                log_infos["performance/train_length"] =  train_traj_lengths[-1]

                if tot_env_steps % self.train_model_interval == 0 and self.model_env_ratio > 0.0:
                    #train model
                    train_model_start_time = time()
                    model_log_infos = self.train_model()
                    train_model_used_time =  time() - train_model_start_time

                    #rollout model
                    rollout_model_start_time = time()
                    self.rollout_model(model_rollout_steps=model_rollout_steps)
                    rollout_model_used_time =  time() - rollout_model_start_time

                    log_infos["times/train_model"] =  train_model_used_time
                    log_infos["times/rollout_model"] =  rollout_model_used_time
                    log_infos.update(model_log_infos)

                #train agent
                train_agent_start_time = time()
                for agent_update_step in range(self.num_agent_updates_per_env_step):
                    agent_log_infos = self.train_agent()
                    tot_agent_update_steps += 1
                train_agent_used_time =  time() - train_agent_start_time

                log_infos.update(agent_log_infos)
                log_infos['times/train_agent'] = train_agent_used_time
                log_infos["misc/utd_ratio"] = tot_agent_update_steps / tot_env_steps
                log_infos["misc/tot_agent_update_steps"] = tot_agent_update_steps

                self.post_iter(log_infos, tot_env_steps)         

            epoch_end_time = time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_durations.append(epoch_duration)
            util.logger.log_var("times/epoch_duration", epoch_duration, tot_env_steps)
            



    def train_model(self):
        #get train and eval data
        num_train_data = int(self.env_buffer.max_sample_size * (1.0 - self.hold_out_ratio))
        env_data = self.env_buffer.sample(self.env_buffer.max_sample_size)
        train_data, eval_data ={}, {}
        for key in env_data.keys():
            train_data[key] = env_data[key][:num_train_data]
            eval_data[key] = env_data[key][num_train_data:]
        self.transition_model.reset_normalizers()
        self.transition_model.update_normalizer(train_data['obs'], train_data['action'])

        #train model
        model_train_iters = 0
        model_train_epochs = 0
        num_epochs_since_prev_best = 0
        break_training = False
        self.transition_model.reset_best_snapshots()
        while not break_training:
            for train_data_batch in dict_batch_generator(train_data, self.model_batch_size):
                model_log_infos= self.transition_model.update(train_data_batch)
                model_train_iters += 1
                self.model_tot_train_timesteps += 1
            
            eval_mse_losses, _ = self.transition_model.eval_data(eval_data, update_elite_models=False)
            util.logger.log_var("loss/model_eval_mse_loss", eval_mse_losses.mean(), self.model_tot_train_timesteps)
            updated = self.transition_model.update_best_snapshots(eval_mse_losses)
            updated = 0.0
            num_epochs_since_prev_best += 1
            if updated > 0.01:
                model_train_epochs += num_epochs_since_prev_best
                num_epochs_since_prev_best = 0
            if num_epochs_since_prev_best >= self.max_model_update_epochs_to_improve or model_train_iters > self.max_model_train_iterations:
                break
        self.transition_model.load_best_snapshots()



        # evaluate data to update the elite models
        self.transition_model.eval_data(eval_data, update_elite_models=True)
        model_log_infos['misc/norm_obs_mean'] = torch.mean(self.transition_model.obs_normalizer.mean).item()
        model_log_infos['misc/norm_obs_var'] = torch.mean(self.transition_model.obs_normalizer.var).item()
        model_log_infos['misc/norm_act_mean'] = torch.mean(self.transition_model.act_normalizer.mean).item()
        model_log_infos['misc/norm_act_var'] = torch.mean(self.transition_model.act_normalizer.var).item()
        model_log_infos['misc/model_train_epochs'] = model_train_epochs
        model_log_infos['misc/model_train_train_steps'] = model_train_iters
        return model_log_infos

    def resize_model_buffer(self, rollout_length):
        rollouts_per_epoch = self.rollout_batch_size * self.num_env_steps_per_epoch / self.train_model_interval
        new_model_buffer_size = int(rollout_length * rollouts_per_epoch * self.model_retain_epochs)

        self.model_buffer.resize(new_model_buffer_size)

    @torch.no_grad()
    def rollout_model(self, model_rollout_steps):
        rollout_data_batch =  self.env_buffer.sample(self.rollout_batch_size, to_tensor=False, allow_duplicate=True)
        obs_batch = rollout_data_batch['obs']
        #perform k-step model rollout starting from s_t using policy\pi
        rollout_batch_nums = int(np.ceil(self.rollout_batch_size / self.rollout_mini_batch_size))
        for rollout_batch_id in range(rollout_batch_nums):
            
            obs_minibatch = obs_batch[rollout_batch_id * self.rollout_mini_batch_size: min(len(obs_batch), (rollout_batch_id + 1) * self.rollout_mini_batch_size)]
            for rollout_step in range(model_rollout_steps):
                action_minibatch = self.agent.select_action(obs_minibatch, deterministic=True)['action']

                next_obs_minibatch, reward_minibatch, done_minibatch = self.transition_model.predict(obs_minibatch, action_minibatch)
                done_minibatch = [float(d) for d in done_minibatch]
                self.model_buffer.add_traj(obs_minibatch, action_minibatch, next_obs_minibatch, reward_minibatch, done_minibatch)
                obs_minibatch = np.array([next_obs_pred for next_obs_pred, done_pred in zip(next_obs_minibatch, done_minibatch) if not done_pred])
                
                if len(obs_minibatch) == 0:
                    break

    
    def train_agent(self):
        train_agent_model_batch_size = int(self.agent_batch_size * self.model_env_ratio)
        train_agent_env_batch_size = self.agent_batch_size - train_agent_model_batch_size
        model_data_batch = self.model_buffer.sample(train_agent_model_batch_size)
        env_data_batch = self.env_buffer.sample(train_agent_env_batch_size)
        data_batch = functional.merge_data_batch(model_data_batch, env_data_batch)
        loss_dict = self.agent.update(data_batch)
        return loss_dict