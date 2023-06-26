from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from code.agents.ddpg.h_ddpg_agent import DdpgAgent
from code.policies import h_actor_policy


@gin.configurable
class HierarchicalDdpgAgent:

    def __init__(self,
                 intention_time_step_spec,
                 intention_action_spec,
                 control_time_step_spec,
                 control_action_spec,
                 intention_actor_network,
                 intention_critic_network,
                 control_actor_network,
                 control_critic_network,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 ou_stddev=1.0,
                 ou_damping=1.0,
                 target_actor_network=None,
                 target_critic_network=None,
                 target_update_tau=1.0,
                 target_update_period=1,
                 dqda_clipping=None,
                 td_errors_loss_fn=None,
                 gamma=1.0,
                 reward_scale_factor=1.0,
                 gradient_clipping=None,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None,
                 name=None):

        self.intention_agent = DdpgAgent(
            intention_time_step_spec,
            intention_action_spec,
            actor_network=intention_actor_network,
            critic_network=intention_critic_network,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            ou_stddev=None,
            ou_damping=None,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            dqda_clipping=dqda_clipping,
            td_errors_loss_fn=td_errors_loss_fn,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars
        )
        self.control_agent = DdpgAgent(
            control_time_step_spec,
            control_action_spec,
            actor_network=control_actor_network,
            critic_network=control_critic_network,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            ou_stddev=ou_stddev,
            ou_damping=ou_damping,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            dqda_clipping=dqda_clipping,
            td_errors_loss_fn=td_errors_loss_fn,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars
        )

        self.policy = h_actor_policy.HierarchicalActorPolicy(
            intention_time_step_spec=intention_time_step_spec,
            intention_action_spec=intention_action_spec,
            control_time_step_spec=control_time_step_spec,
            control_action_spec=control_action_spec,
            intention_agent=self.intention_agent,
            control_agent=self.control_agent,
            clip=True
        )

        self.collect_policy = h_actor_policy.HierarchicalActorPolicy(
            intention_time_step_spec=intention_time_step_spec,
            intention_action_spec=intention_action_spec,
            control_time_step_spec=control_time_step_spec,
            control_action_spec=control_action_spec,
            intention_agent=self.intention_agent,
            control_agent=self.control_agent,
            clip=False
        )
