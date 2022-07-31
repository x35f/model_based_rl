overwrite_args = {
  "env_name": "AntTruncatedObs-v2",
  "trainer": {
    "num_agent_updates_per_env_step": 20,
    "max_epoch": 300
  },
  "rollout_step_scheduler":{
    "initial_val": 1,
    "target_val": 25,
    "start_timestep": 20,
    "end_timestep": 100,
    "schedule_type": "linear"
  },
  "agent": {
    "entropy": {
      "target_entropy": -4
    }
  }
}
