from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
from tf_agents.policies import actor_policy

from code.utils.common_utils import intention_value, duplicate_digits_2


@gin.configurable
class HierarchicalActorPolicy(actor_policy.ActorPolicy):

    def __init__(self,
                 intention_time_step_spec,
                 intention_action_spec,
                 control_time_step_spec,
                 control_action_spec,
                 intention_agent,
                 control_agent,
                 info_spec=(),
                 clip=True,
                 training=False,
                 name=None):
        self.intention_agent = intention_agent
        self.control_agent = control_agent
        self.intention_time_step_spec = intention_time_step_spec
        self.intention_action_spec = intention_action_spec
        self.control_time_step_spec = control_time_step_spec
        self.control_action_spec = control_action_spec

        super(HierarchicalActorPolicy, self).__init__(
            time_step_spec=intention_time_step_spec,
            action_spec=control_action_spec,
            actor_network=intention_agent._actor_network,
            info_spec=info_spec,
            clip=clip,
            name=name)

    def _action(self, time_step, policy_state, seed):
        intention_step = self.intention_agent.policy.action(time_step, policy_state)
        intention_step = intention_step._replace(action=(duplicate_digits_2(intention_step.action)))
        intention = intention_value(intention_step.action)
        time_step.observation.update({'intention': intention})
        action_step = self.control_agent.policy.action(time_step, policy_state)
        return action_step

