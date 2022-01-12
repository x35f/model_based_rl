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
    def __init__(self, agent, env, eval_env, transition_model, env_buffer, model_buffer, rollout_step_generator,
            agent_batch_size=100,
            model_batch_size=256,
            rollout_batch_size=100000,
            rollout_mini_batch_size=1000,
            model_retain_epochs=1,
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
            max_model_update_epochs_to_improve=5,
            start_train_model_timestep=5000,
            start_train_agent_timestep = 10000,
            save_video_demo_interval=10000,
            log_interval=100,
            model_env_ratio=0.8,
            hold_out_ratio=0.1,
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
        self.model_retain_epochs = model_retain_epochs
        self.num_env_steps_per_epoch = num_env_steps_per_epoch
        self.train_agent_interval = train_agent_interval
        self.train_model_interval = train_model_interval
        self.num_agent_updates_per_env_step = num_agent_updates_per_env_step
        self.max_agent_updates_per_env_step = max_agent_updates_per_env_step
        self.max_model_update_epochs_to_improve = max_model_update_epochs_to_improve
        self.max_trajectory_length = max_trajectory_length
        self.test_interval = test_interval
        self.num_test_trajectories = num_test_trajectories
        self.max_epoch = max_epoch
        self.save_model_interval = save_model_interval
        self.save_video_demo_interval = save_video_demo_interval
        self.start_train_model_timestep = start_train_model_timestep
        self.start_train_agent_timestep = start_train_agent_timestep
        self.log_interval = log_interval
        self.model_env_ratio = model_env_ratio
        self.hold_out_ratio = hold_out_ratio
        self.train_log_interval = train_log_interval
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load(load_dir)
        self.model_tot_train_timesteps = 0

    def warmup(self):
        obs = self.env.reset()
        for step in tqdm(range(self.start_train_model_timestep)):
            action, _ = self.agent.select_action(obs)
            next_obs, reward, done, info = self.env.step(action)
            self.env_buffer.add_tuple(obs, action, next_obs, reward, float(done))
            obs = next_obs
            if done:
                obs = self.env.reset()



    def train(self):
        train_traj_returns = [0]
        train_traj_lengths = [0]
        epoch_durations = []
        traj_return = 0
        traj_length = 0
        util.logger.log_str("Warming Up")
        self.warmup()
        self.tot_env_steps = self.start_train_model_timestep
        
        tot_agent_update_steps = 0
        model_rollout_steps = int(self.rollout_step_generator.initial_val)
        self.resize_model_buffer(model_rollout_steps)
        model_loss_dict = {}
        loss_dict = {}
        obs = self.env.reset()
        done = False
        util.logger.log_str("Started Training")
        for epoch in trange(self.max_epoch, colour='blue', desc='outer loop'): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            
            epoch_start_time = time()
            prev_model_rollout_steps = model_rollout_steps
            model_rollout_steps = int(self.rollout_step_generator.next())
            if prev_model_rollout_steps != model_rollout_steps:
                self.resize_model_buffer(model_rollout_steps)

            #train model on env_buffer via maximum likelihood
            #for e steps do
            for env_step in trange(self.num_env_steps_per_epoch, colour='green', desc='inner loop'):
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

                if self.tot_env_steps % self.train_model_interval == 0 and self.model_env_ratio > 0.0:
                    #train model
                    train_model_start_time = time()
                    model_loss_dict = self.train_model()

                    train_model_used_time =  time() - train_model_start_time

                    #rollout model
                    rollout_model_start_time = time()
                    self.rollout_model(model_rollout_steps=model_rollout_steps)
                    rollout_model_used_time =  time() - rollout_model_start_time

                    for loss_name in model_loss_dict:
                        util.logger.log_var(loss_name, model_loss_dict[loss_name], self.tot_env_steps) 
                    util.logger.log_var("times/model_train", train_model_used_time, self.tot_env_steps)
                    util.logger.log_var("times/rollout_model", rollout_model_used_time, self.tot_env_steps)

                train_agent_start_time = time()
                #train agent
                if self.tot_env_steps > self.start_train_agent_timestep:
                    for agent_update_step in range(self.num_agent_updates_per_env_step) :
                        if tot_agent_update_steps > self.tot_env_steps * self.max_agent_updates_per_env_step:
                            break
                        loss_dict = self.train_agent()
                        tot_agent_update_steps += 1

                train_agent_used_time =  time() - train_agent_start_time
                       
                if env_step % self.train_log_interval == 0:
                    util.logger.log_var("times/train_agent", train_agent_used_time, self.tot_env_steps)
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



    def train_model(self):
        num_train_data = int( self.env_buffer.max_sample_size * (1.0 - self.hold_out_ratio) )
        env_data = self.env_buffer.sample(self.env_buffer.max_sample_size)

    
        train_data, eval_data ={}, {}
        for key in env_data.keys():
            train_data[key] = env_data[key][:num_train_data]
            eval_data[key] = env_data[key][num_train_data:]
        self.transition_model.reset_normalizers()
        self.transition_model.update_normalizer(train_data['obs'], train_data['action'])
        #train model
        prev_best_loss = self.transition_model.eval_data(eval_data)
        model_train_iters = 0
        model_train_epochs = 0
        num_epochs_since_prev_best = 0
        break_training = False
        self.transition_model.reset_best_snapshots()
        while not break_training:
            for train_data_batch in dict_batch_generator(train_data, self.model_batch_size):
                model_loss_dict = self.transition_model.update(train_data_batch)
                model_train_iters += 1
                self.model_tot_train_timesteps += 1
            
            model_train_epochs += 1

            eval_mse_losses, _ = self.transition_model.eval_data(eval_data)
            util.logger.log_var("model/mse_loss", eval_mse_losses.mean(), self.model_tot_train_timesteps)
            updated = self.transition_model.update_best_snapshots(eval_mse_losses)
            if updated > 0.01:
                num_epochs_since_prev_best = 0
            else:
                num_epochs_since_prev_best += 1
            if num_epochs_since_prev_best >= self.max_model_update_epochs_to_improve:
                break
        self.transition_model.load_best_snapshots()



        # evaluate data to update the elite models
        self.transition_model.select_elite_models(eval_data)
        model_loss_dict['misc/norm_obs_mean'] = torch.mean(self.transition_model.obs_normalizer.mean).item()
        model_loss_dict['misc/norm_obs_var'] = torch.mean(self.transition_model.obs_normalizer.var).item()
        model_loss_dict['misc/norm_act_mean'] = torch.mean(self.transition_model.act_normalizer.mean).item()
        model_loss_dict['misc/norm_act_var'] = torch.mean(self.transition_model.act_normalizer.var).item()
        model_loss_dict['model/num_epochs'] = model_train_epochs
        model_loss_dict['model/num_train_steps'] = model_train_iters
        return model_loss_dict

    def resize_model_buffer(self, rollout_length):
        rollouts_per_epoch = self.rollout_batch_size * self.num_env_steps_per_epoch / self.train_model_interval
        new_model_buffer_size = int(rollout_length * rollouts_per_epoch * self.model_retain_epochs)

        self.model_buffer.resize(new_model_buffer_size)
        util.logger.log_var("misc/model_buffer_size", new_model_buffer_size, self.tot_env_steps)

    @torch.no_grad()
    def rollout_model(self, model_rollout_steps):
        rollout_data_batch =  self.env_buffer.sample(self.rollout_batch_size, to_tensor=False, allow_duplicate=True)
        obs_batch = rollout_data_batch['obs']
        #perform k-step model rollout starting from s_t using policy\pi
        rollout_batch_nums = int(np.ceil(self.rollout_batch_size / self.rollout_mini_batch_size))
        for rollout_batch_id in range(rollout_batch_nums):
            
            obs_minibatch = obs_batch[rollout_batch_id * self.rollout_mini_batch_size: min(len(obs_batch), (rollout_batch_id + 1) * self.rollout_mini_batch_size)]
            for rollout_step in range(model_rollout_steps):
                action_minibatch, _ = self.agent.select_action(obs_minibatch, deterministic=True)

                next_obs_minibatch, reward_minibatch, done_minibatch, info = self.transition_model.predict(obs_minibatch, action_minibatch)
                done_minibatch = [float(d) for d in done_minibatch]
                self.model_buffer.add_traj(obs_minibatch, action_minibatch, next_obs_minibatch, reward_minibatch, done_minibatch)
                obs_minibatch = np.array([next_obs_pred for next_obs_pred, done_pred in zip(next_obs_minibatch, done_minibatch) if not done_pred])
                
                if len(obs_minibatch) == 0:
                    break

        util.logger.log_var("misc/env_obs_mean", np.mean(obs_batch), self.tot_env_steps)
        util.logger.log_var("misc/env_obs_var", np.var(obs_batch), self.tot_env_steps)

    
    def train_agent(self):
        train_agent_model_batch_size = int(self.agent_batch_size * self.model_env_ratio)
        train_agent_env_batch_size = self.agent_batch_size - train_agent_model_batch_size
        model_data_batch = self.model_buffer.sample(train_agent_model_batch_size)
        env_data_batch = self.env_buffer.sample(train_agent_env_batch_size)
        data_batch = functional.merge_data_batch(model_data_batch, env_data_batch)
        loss_dict = self.agent.update(data_batch)
        return loss_dict

            
    @torch.no_grad()
    def test(self):
        rewards = []
        lengths = []
        for episode in range(self.num_test_trajectories):
            traj_reward = 0
            traj_length = 0
            state = self.eval_env.reset()
            for step in range(self.max_trajectory_length):
                action, _ = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, _ = self.eval_env.step(action)
                traj_reward += reward
                state = next_state
                traj_length += 1 
                if done:
                    break
            lengths.append(traj_length)
            rewards.append(traj_reward)
        return {
            "return/test": np.mean(rewards),
            "length/test": np.mean(lengths)
        }

