from gymnasium.envs.registration import register
from psyonic_playing_xylophone.envs.psyonic_thumb import PsyonicThumbEnv
from psyonic_playing_xylophone.envs.psyonic_thumb_wrist import PsyonicThumbWristEnv
from psyonic_playing_xylophone.envs.psyonic_thumb_real import PsyonicThumbRealEnv

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

register(
    id="PsyonicThumbRealEnv-v0",
    entry_point="psyonic_playing_xylophone.envs.psyonic_thumb_real:PsyonicThumbRealEnv",
    max_episode_steps=100,
)

register(
    id="PsyonicThumbWristDoubleEnv-v0",
    entry_point="psyonic_playing_xylophone.envs.thumb_wrist_double_hit:PsyonicThumbWristDoubleEnv",
    max_episode_steps=100,
)

register(
    id="PsyonicThumbDoubleEnv-v0",
    entry_point="psyonic_playing_xylophone.envs.thumb_double_hit:PsyonicThumbDoubleEnv"
)