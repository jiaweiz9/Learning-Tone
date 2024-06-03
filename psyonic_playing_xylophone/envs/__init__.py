from gymnasium.envs.registration import register
from psyonic_playing_xylophone.envs.psyonic_thumb import PsyonicThumbEnv

register(
     id="PsyonicThumbEnv-v0",
     entry_point="psyonic_playing_xylophone.envs.psyonic_thumb:PsyonicThumbEnv",
     max_episode_steps=100,
)
