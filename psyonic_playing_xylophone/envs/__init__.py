from gymnasium.envs.registration import register
from psyonic_playing_xylophone.envs.psyonic_thumb import PsyonicThumbEnv
from psyonic_playing_xylophone.envs.psyonic_thumb_wrist import PsyonicThumbWristEnv

register(
     id="PsyonicThumbEnv-v0",
     entry_point="psyonic_playing_xylophone.envs.psyonic_thumb:PsyonicThumbEnv",
     max_episode_steps=100,
)

register(
    id="PsyonicThumbWristEnv-v0",
    entry_point="psyonic_playing_xylophone.envs.psyonic_thumb_wrist:PsyonicThumbWristEnv",
    max_episode_steps=100,
)

register(
    id="PsyonicThumbWristRealEnv-v0",
    entry_point="psyonic_playing_xylophone.envs.psyonic_thumb_wrist_real:PsyonicThumbWristRealEnv",
    max_episode_steps=100,
)