from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from code.policies import latent_actor_policy


@gin.configurable
class HierarchicalLatentActorPolicy(latent_actor_policy.LatentActorPolicy):

    def __init__(self,
                 intention_time_step_spec,
                 intention_action_spec,
                 control_time_step_spec,
                 control_action_spec,
                 intention_agent,
                 control_agent,
                 clip=True,
                 name=None):
        self.intention_agent = intention_agent
        self.control_agent = control_agent
        self.intention_time_step_spec = intention_time_step_spec
        self.intention_action_spec = intention_action_spec
        self.control_time_step_spec = control_time_step_spec
        self.control_action_spec = control_action_spec

        super(HierarchicalLatentActorPolicy, self).__init__(
            time_step_spec=intention_time_step_spec,
            action_spec=control_action_spec,
            inner_policy=intention_agent.inner_policy,
            model_network=intention_agent.model_network,
            clip=clip,
            name=name)

    def _action(self, time_step, policy_state, seed):
        intention_step = self.intention_agent.policy.action(time_step, policy_state)
        control_time_step = time_step.observation.append(intention_step.action)
        action_step = self.control_agent.policy.action(control_time_step, policy_state)
        return action_step
