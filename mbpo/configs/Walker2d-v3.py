overwrite_args = {
  "env_name": "Walker2d-v3",
  "trainer": {
    "max_epoch": 300,
    "num_agent_updates_per_env_step": 20,
  },
  "rollout_step_scheduler":{
    "initial_val": 1,
    "target_val": 1,
    "schedule_type": "identical"
  },
  "agent": {
    "entropy": {
      "automatic_tuning": True,
      "target_entropy": -3
    }
  }
}

