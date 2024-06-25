from stable_baselines3.common.callbacks import BaseCallback

class RewardThresholdCallback(BaseCallback):
    def __init__(self, verbose: int = 0, th_update_freq = 10000):
        self.th_update_freq = th_update_freq
        super().__init__(verbose)
    
    def _on_rollout_start(self) -> None:
        print(f"Rollout {self.num_timesteps // self.th_update_freq} started")
        self.training_env.set_attr("iteration", self.num_timesteps // self.th_update_freq, 0)
        return super()._on_rollout_start()
    