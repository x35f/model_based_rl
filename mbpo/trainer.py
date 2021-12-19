from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
import cv2
import os
from tqdm import  trange
import torch
import random
from unstable_baselines.common import util
from unstable_baselines.common.scheduler import Scheduler

class MBPOTrainer(BaseTrainer):
    def __init__(self, agent, env, eval_env, transition_model, env_buffer, model_buffer, rollout_step_generator,
            agent_batch_size=100,
            model_batch_size=256,
            rollout_batch_size=100000,
            rollout_mini_batch_size=1000,
            num_env_steps_per_epoch=1000,
            train_model_interval=250,
            train_agent_interval=1,
            num_agent_updates_per_env_step=20, # G
            num_model_rollouts=100, #  M
            max_trajectory_length=1000,
            test_interval=10,
            num_test_trajectories=5,
            max_epoch=100000,
            save_model_interval=10000,
            max_agent_updates_per_env_step=5,
            start_timestep=5000,
            save_video_demo_interval=10000,
            log_interval=100,
            model_env_ratio=0.8,
            load_dir="",
            train_log_interval=100,
            **kwargs):
        self.agent = agent
        self.env_buffer = env_buffer
        self.model_buffer = model_buffer
        self.env = env 
        self.eval_env = eval_env
        self.transition_model = transition_model
        self.rollout_step_generator = rollout_step_generator 
        #hyperparameters
        self.num_model_rollouts = num_model_rollouts 
        self.agent_batch_size = agent_batch_size
        self.model_batch_size = model_batch_size
        self.rollout_batch_size = rollout_batch_size
        self.rollout_mini_batch_size = rollout_mini_batch_size
        self.num_env_steps_per_epoch = num_env_steps_per_epoch
        self.train_agent_interval = train_agent_interval
        self.train_model_interval = train_model_interval
        self.num_agent_updates_per_env_step = num_agent_updates_per_env_step
        self.max_agent_updates_per_env_step = max_agent_updates_per_env_step
        self.max_trajectory_length = max_trajectory_length
        self.test_interval = test_interval
        self.num_test_trajectories = num_test_trajectories
        self.max_epoch = max_epoch
        self.save_model_interval = save_model_interval
        self.save_video_demo_interval = save_video_demo_interval
        self.start_timestep = start_timestep
        self.log_interval = log_interval
        self.model_env_ratio = model_env_ratio
        self.train_log_interval = train_log_interval
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load(load_dir)

    def train(self):
        train_traj_returns = [0]
        train_traj_lengths = [0]
        epoch_durations = []
        self.tot_env_steps = 0
        obs = self.env.reset()
        traj_return = 0
        traj_length = 0
        done = False
        obs = self.env.reset()
        tot_agent_update_steps = 0
        model_loss_dict = {}
        loss_dict = {}
        for epoch in trange(self.max_epoch, colour='red', desc='outer loop'): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            
            epoch_start_time = time()
            model_rollout_steps = int(self.rollout_step_generator.next())

            #train model on env_buffer via maximum likelihood
            #for e steps do
            for env_step in trange(self.num_env_steps_per_epoch, colour='red', desc='inner loop'):
                #take action in environment according to \pi, add to D_env
                action, _ = self.agent.select_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                traj_length  += 1
                traj_return += reward
                if traj_length == self.max_trajectory_length:
                    done = False # for mujoco env
                self.env_buffer.add_tuple(obs, action, next_obs, reward, float(done))
                obs = next_obs
                if done or traj_length >= self.max_trajectory_length:
                    obs = self.env.reset()
                    train_traj_returns.append(traj_return)
                    train_traj_lengths.append(traj_length)
                    util.logger.log_var("return/train",traj_return , self.tot_env_steps)
                    util.logger.log_var("length/train",traj_length, self.tot_env_steps)
                    traj_length = 0
                    traj_return = 0
                self.tot_env_steps += 1
                if self.tot_env_steps <= self.start_timestep:
                    continue

                if self.tot_env_steps % self.train_model_interval == 0 and self.model_env_ratio > 0.0:
                    #train model
                    model_loss_dict = self.train_model()

                    #rollout model
                    self.rollout_model(model_rollout_steps=model_rollout_steps)

                for loss_name in model_loss_dict:
                    util.logger.log_var(loss_name, model_loss_dict[loss_name], self.tot_env_steps) 

                train_agent_start_time = time()
                #for G gradient updates do
                for agent_update_step in range(self.num_agent_updates_per_env_step) :
                    if tot_agent_update_steps > self.tot_env_steps * self.max_agent_updates_per_env_step:
                        break
                    model_data_batch = self.model_buffer.sample_batch(int(self.agent_batch_size * self.model_env_ratio))
                    env_data_batch = self.env_buffer.sample_batch(int(self.agent_batch_size * (1.0 - self.model_env_ratio)))
                    data_batch = self.merge_data_batch(model_data_batch, env_data_batch)
                    loss_dict = self.agent.update(data_batch)
                    tot_agent_update_steps += 1
                train_agent_used_time =  time() - train_agent_start_time
                util.logger.log_var("times/train_agent", train_agent_used_time, self.tot_env_steps)
                       
                if env_step % self.train_log_interval == 0:
                    for loss_name in loss_dict:
                        util.logger.log_var(loss_name, loss_dict[loss_name], self.tot_env_steps)
            

            epoch_end_time = time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_durations.append(epoch_duration)
            if epoch % self.log_interval == 0:
                util.logger.log_var("misc/utd_ratio", tot_agent_update_steps / self.tot_env_steps, self.tot_env_steps)
                util.logger.log_var("misc/tot_agent_update_steps", tot_agent_update_steps, self.tot_env_steps)
                util.logger.log_var("misc/model_rollout_steps", model_rollout_steps, self.tot_env_steps)
                if self.model_env_ratio> 0.0:
                    for loss_name in model_loss_dict:
                        util.logger.log_var(loss_name, model_loss_dict[loss_name], self.tot_env_steps)
                for loss_name in loss_dict:
                    util.logger.log_var(loss_name, loss_dict[loss_name], self.tot_env_steps)
            if epoch % self.test_interval == 0:
                test_start_time = time()
                log_dict = self.test()
                test_used_time = time() - test_start_time
                avg_test_reward = log_dict['return/test']
                for log_key in log_dict:
                    util.logger.log_var(log_key, log_dict[log_key], self.tot_env_steps)
                util.logger.log_var("times/test", test_used_time, self.tot_env_steps)
                remaining_seconds = int((self.max_epoch - epoch + 1) * np.mean(epoch_durations[-100:]))
                time_remaining_str = util.second_to_time_str(remaining_seconds)
                summary_str = "iteration {}/{}:\ttrain return {:.02f}\ttest return {:02f}\teta: {}".format(epoch, self.max_epoch, train_traj_returns[-1],avg_test_reward,time_remaining_str)
                util.logger.log_str(summary_str)
            if epoch % self.save_model_interval == 0:
                self.agent.save_model(epoch)
                self.transition_model.save_model(epoch)
            if self.save_video_demo_interval > 0 and epoch % self.save_video_demo_interval == 0:
                self.save_video_demo(epoch)

    def merge_data_batch(self, data1_dict, data2_dict):
        for key in data1_dict:
            if isinstance(data1_dict[key], np.ndarray):
                data1_dict[key]  = np.concatenate([data1_dict[key], data2_dict[key]], axis=0)
            elif isinstance(data1_dict[key], torch.Tensor):
                data1_dict[key]  = torch.cat([data1_dict[key], data2_dict[key]], dim=0)
        return data1_dict

    def train_model(self):
        train_model_start_time = time()
        data_indices = list(range(self.env_buffer.max_sample_size))
        random.shuffle(data_indices)
        num_batches = int(np.ceil(self.env_buffer.max_sample_size / self.model_batch_size))
        for model_train_step in range(num_batches):
            batch_indices = data_indices[model_train_step * self.model_batch_size: min(len (data_indices), (model_train_step + 1) * self.model_batch_size)]
            data_batch = self.env_buffer.get_batch(batch_indices)
            model_loss_dict = self.transition_model.update(data_batch)
        train_model_used_time = time() - train_model_start_time
        #print("\033[32menv\033[0m", data_batch['done'].shape, data_batch['done'][:min(3, len(data_batch['done']))])
        return model_loss_dict


    def rollout_model(self, model_rollout_steps):
        train_model_data_batch =  self.env_buffer.sample_batch(self.rollout_batch_size, to_tensor=False, allow_duplicate=True)
        obs_batch = train_model_data_batch['obs']
        next_obs_batch = train_model_data_batch['next_obs']
        #perform k-step model rollout starting from s_t using policy\pi
        rollout_batch_nums = int(np.ceil(self.rollout_batch_size / self.rollout_mini_batch_size))
        for rollout_batch_id in range(rollout_batch_nums):
            
            obs_minibatch = obs_batch[rollout_batch_id * self.rollout_mini_batch_size: min(len(obs_batch), (rollout_batch_id + 1) * self.rollout_mini_batch_size)]

            for rollout_step in range(model_rollout_steps):
                action_minibatch, _ = self.agent.select_action(obs_minibatch)
                next_obs_minibatch, reward_minibatch, done_minibatch, info = self.transition_model.predict(obs_minibatch, action_minibatch)
                done_minibatch = [float(d) for d in done_minibatch]
                #print("\033[31mrollout\033[0m", done_minibatch.shape, done_minibatch[:min(3, len(done_minibatch))])
                self.model_buffer.add_traj(obs_minibatch, action_minibatch, next_obs_minibatch, reward_minibatch, done_minibatch)
                #print(obs_minibatch.shape, action_minibatch.shape, next_obs_minibatch.shape, reward_minibatch.shape, done_minibatch.shape)
                obs_minibatch = np.array([obs_pred for obs_pred, done_pred in zip(next_obs_minibatch, done_minibatch) if not done_pred])
                if len(obs_minibatch) == 0:
                    break


        util.logger.log_var("misc/env_obs_mean", np.mean(obs_batch), self.tot_env_steps)
        util.logger.log_var("misc/env_obs_var", np.var(obs_batch), self.tot_env_steps)
        util.logger.log_var("misc/env_next_obs_mean", np.mean(next_obs_batch), self.tot_env_steps)
        util.logger.log_var("misc/env_next_obs_var", np.var(next_obs_batch), self.tot_env_steps)


            


