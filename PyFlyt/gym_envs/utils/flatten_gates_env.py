"""Wrapper class for flattening the gate envs to use homogeneous observation spaces."""
from __future__ import annotations

import numpy as np
from gymnasium.core import Env, ObservationWrapper
from gymnasium.spaces import Box


class FlattenGatesEnv(ObservationWrapper):
    """Wrapper class to flatten the observation space of the QuadXGatesEnv environment."""

    def __init__(self, env: Env, context_length: int = 2):
        """
        Initializes the FlattenGatesEnv wrapper.

        Args:
        ----
            env (Env): A PyFlyt Gates environment.
            context_length (int): Number of gates to include in the flattened observation space.
        """
        super().__init__(env=env)
        self.context_length = context_length
        self.attitude_shape = env.observation_space["attitude"].shape[0]
        self.target_shape = env.observation_space["target_deltas"].feature_space.shape[0]
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.attitude_shape + self.target_shape * self.context_length,),
        )

    def observation(self, observation) -> np.ndarray:
        """
        Flattens an observation from the QuadXGatesEnv environment.

        Args:
        ----
            observation: A dictionary observation with "attitude", "rgba_cam", and "target_deltas" keys.

        Returns:
        -------
            np.ndarray: The flattened observation.
        """
        num_targets = min(self.context_length, observation["target_deltas"].shape[0])

        targets = np.zeros((self.context_length, self.target_shape))
        targets[:num_targets] = observation["target_deltas"][:num_targets]

        new_obs = np.concatenate([observation["attitude"], *targets])

        return new_obs
