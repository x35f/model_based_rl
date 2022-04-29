overwrite_args = {
  "env_name": "HumanoidTruncatedObs-v2",
  "trainer": {

    "train_model_interval": 1000,
    "max_epoch": 300,
    "num_agent_updates_per_env_step": 20,
    "model_retain_epochs": 5
  },
  "rollout_step_scheduler":{
    "initial_val": 1,
    "target_val": 25,
    "start_timestep": 20,
    "end_timestep": 300,
    "schedule_type": "linear"
  },
  "agent": {
    "entropy": {
      "target_entropy": -2
    }
  },
  "transition_model":{
    "model":{
      "hidden_dims": [400, 400, 400, 400]
    }
  }
}
