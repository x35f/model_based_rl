overwrite_args = {
  "env_name": "InvertedPendulum-v2",
  "trainer": {
    "max_epoch": 80,
    "eval_interval": 250,
    "save_video_demo_interval": -1,
    "num_env_steps_per_epoch": 250,
    "warmup_timesteps": 500,
    "num_agent_updates_per_env_step": 10,
    "max_trajectory_length":1000
  },
  "rollout_step_scheduler":{
    "initial_val": 1,
    "target_val": 1,
    "schedule_type": "identical"
  },
  "agent": {
    "entropy": {
      "automatic_tuning": True,
      "target_entropy": -0.05
    }
  }
}
