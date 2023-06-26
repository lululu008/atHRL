from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.policies import actor_policy


@gin.configurable
class TrajectoryActorPolicy(actor_policy.ActorPolicy):
    def __init__(self,
        time_step_spec,
        action_spec,
        actor_network,
        info_spec=(),
        observation_normalizer=None,
        clip=True,
        training=False,
        name=None):

        super(TrajectoryActorPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=actor_network,
            info_spec=info_spec,
        clip=clip,
            name=name)

    def trajectory_action(self, time_step, policy_state, seed):
        
        return 

