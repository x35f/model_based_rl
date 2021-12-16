from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
import cv2
import os
from tqdm import  tqdm
import torch
from unstable_baselines.common import util
from unstable_baselines.common.scheduler import Scheduler

class MBPOTrainer(BaseTrainer):
    def __init__(self, agent, env, eval_env, env_buffer, model_buffer, rollout_step_scheduler,
            agent_batch_size=100,
            model_batch_size=256,
            rollout_batch_size=100000,
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
            start_timestep=1000,
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
        self.rollout_step_generator = Scheduler(**rollout_step_scheduler)
        #hyperparameters
        self.num_model_rollouts = num_model_rollouts 
        self.agent_batch_size = agent_batch_size
        self.model_batch_size = model_batch_size
        self.rollout_batch_size = rollout_batch_size
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

    def warmup(self, warmup_steps=2000):
        #add warmup transitions to buffer
        obs = self.env.reset()
        traj_length = 0
        traj_reward = 0
        for step in range(warmup_steps):
            action, _ = self.agent.select_action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            traj_length  += 1
            traj_reward += reward
            if traj_length >= self.max_trajectory_length - 1:
                done = True
            if self.agent.per:
                self.env_buffer.add_tuple(obs, action, next_obs, reward, float(done), self.buffer.max)
            else:
                self.env_buffer.add_tuple(obs, action, next_obs, reward, float(done))
            obs = next_obs
            if done or traj_length >= self.max_trajectory_length:
                obs = self.env.reset()
                traj_length = 0
                traj_reward = 0

    def train(self):
        train_traj_rewards = [0]
        train_traj_lengths = [0]
        epoch_durations = []
        self.warmup(self.start_timestep)
        tot_env_steps = self.start_timestep
        obs = self.env.reset()
        traj_reward = 0
        traj_length = 0
        done = False
        obs = self.env.reset()
        tot_agent_update_steps = 0
        for epoch in tqdm(range(self.max_epoch)): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            
            epoch_start_time = time()
            model_rollout_steps = int(self.rollout_step_generator.next())

            #train model on env_buffer via maximum likelihood
            #for e steps do
            for env_step in range(self.num_env_steps_per_epoch):
                #take action in environment according to \pi, add to D_env
                action, _ = self.agent.select_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                traj_length  += 1
                traj_reward += reward
                if traj_length == self.max_trajectory_length:
                    done = False # for mujoco env
                self.env_buffer.add_tuple(obs, action, next_obs, reward, float(done))
                obs = next_obs
                if done or traj_length >= self.max_trajectory_length:
                    obs = self.env.reset()
                    train_traj_rewards.append(traj_reward)
                    train_traj_lengths.append(traj_length)
                    util.logger.log_var("return/train",traj_reward , tot_env_steps)
                    util.logger.log_var("length/train",traj_length, tot_env_steps)
                    traj_length = 0
                    traj_reward = 0
                tot_env_steps += 1

                if tot_env_steps % self.train_model_interval == 0 and self.model_env_ratio> 0.0:
                    #train model
                    train_model_start_time = time()
                    num_batches = int(np.ceil(self.env_buffer.max_sample_size / self.model_batch_size))
                    for model_train_step in range(num_batches):
                        data_batch = self.env_buffer.get_batch(model_train_step * self.model_batch_size, self.model_batch_size)
                        model_loss_dict = self.agent.train_model(data_batch)
                    train_model_used_time = time() - train_model_start_time
                    for loss_name in model_loss_dict:
                        util.logger.log_var(loss_name, model_loss_dict[loss_name], tot_env_steps) 

                    #sample s_t uniformly from D_env
                    rollout_start_time = time()
                    obs_batch= self.env_buffer.sample_batch(self.rollout_batch_size, allow_duplicate=True)['obs']
                    #perform k-step model rollout starting from s_t using policy\pi
                    generated_transitions = self.agent.rollout(obs_batch, model_rollout_steps)
                    #add the transitions to D_model
                    self.model_buffer.add_traj(**generated_transitions)
                    rollout_used_time = time() - rollout_start_time
                    util.logger.log_var("times/train_model", train_model_used_time, tot_env_steps)
                    util.logger.log_var("times/rollout", rollout_used_time, tot_env_steps)

                train_agent_start_time = time()
                #for G gradient updates do
                for agent_update_step in range(self.num_agent_updates_per_env_step):
                    if tot_agent_update_steps > tot_env_steps * self.max_agent_updates_per_env_step:
                        break
                    model_data_batch = self.model_buffer.sample_batch(int(self.agent_batch_size * self.model_env_ratio))
                    env_data_batch = self.env_buffer.sample_batch(int(self.agent_batch_size * (1 - self.model_env_ratio)))
                    policy_loss_dict_model = self.agent.update(model_data_batch)
                    policy_loss_dict_env = self.agent.update(env_data_batch)
                    tot_agent_update_steps += self.num_agent_updates_per_env_step
                train_agent_used_time =  time() - train_agent_start_time
                util.logger.log_var("times/train_agent", train_agent_used_time, tot_env_steps)
                       
                if env_step % self.train_log_interval == 0:
                    for loss_name in policy_loss_dict_model:
                        util.logger.log_var(loss_name+"_model", policy_loss_dict_model[loss_name], tot_env_steps)
                    for loss_name in policy_loss_dict_env:
                        util.logger.log_var(loss_name+"_env", policy_loss_dict_env[loss_name], tot_env_steps)

            epoch_end_time = time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_durations.append(epoch_duration)
            if epoch % self.log_interval == 0:
                util.logger.log_var("misc/utd_ratio", tot_agent_update_steps / tot_env_steps, tot_env_steps)
                util.logger.log_var("misc/tot_agent_update_steps", tot_agent_update_steps, tot_env_steps)
                util.logger.log_var("misc/model_rollout_steps", model_rollout_steps, tot_env_steps)
                for loss_name in model_loss_dict:
                    util.logger.log_var(loss_name, model_loss_dict[loss_name], tot_env_steps)
                for loss_name in policy_loss_dict_model:
                    util.logger.log_var(loss_name+"_model", policy_loss_dict_model[loss_name], tot_env_steps)
                for loss_name in policy_loss_dict_env:
                    util.logger.log_var(loss_name+"_env", policy_loss_dict_env[loss_name], tot_env_steps)
            if epoch % self.test_interval == 0:
                test_start_time = time()
                log_dict = self.test()
                test_used_time = time() - test_start_time
                avg_test_reward = log_dict['return/test']
                for log_key in log_dict:
                    util.logger.log_var(log_key, log_dict[log_key], tot_env_steps)
                util.logger.log_var("times/test", test_used_time, tot_env_steps)
                remaining_seconds = int((self.max_epoch - epoch + 1) * np.mean(epoch_durations[-100:]))
                time_remaining_str = util.second_to_time_str(remaining_seconds)
                summary_str = "iteration {}/{}:\ttrain return {:.02f}\ttest return {:02f}\teta: {}".format(epoch, self.max_epoch, train_traj_rewards[-1],avg_test_reward,time_remaining_str)
                util.logger.log_str(summary_str)
            if epoch % self.save_model_interval == 0:
                self.agent.save_model(epoch)
            if epoch % self.save_video_demo_interval == 0:
                self.save_video_demo(epoch)


    def test(self):
        rewards = []
        lengths = []
        for episode in range(self.num_test_trajectories):
            traj_reward = 0
            traj_length = 0
            obs = self.eval_env.reset()
            for step in range(self.max_trajectory_length):
                action, _ = self.agent.select_action(obs, deterministic=True)
                next_obs, reward, done, _ = self.eval_env.step(action)
                traj_reward += reward
                obs = next_obs
                traj_length += 1 
                if done:
                    break
            lengths.append(traj_length)
            rewards.append(traj_reward)
        return {
            "return/test": np.mean(rewards),
            "length/test": np.mean(lengths)
        }




            


