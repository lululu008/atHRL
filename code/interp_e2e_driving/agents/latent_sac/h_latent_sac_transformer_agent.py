from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from interp_e2e_driving.agents.latent_sac.latent_sac_transformer_agent import LatentSACAgent
from interp_e2e_driving.policies import h_latent_actor_policy


@gin.configurable
class HierarchicalLatentSACAgent:

    def __init__(self,
                 intention_time_step_spec,
                 intention_action_spec,
                 control_time_step_spec,
                 control_action_spec,
                 inner_agent,
                 model_network,
                 model_optimizer,
                 model_batch_size=None,
                 num_images_per_summary=1,
                 sequence_length=2,
                 gradient_clipping=None,
                 summarize_grads_and_vars=False,
                 train_step_counter=None,
                 fps=10,
                 name=None):

        self.intention_agent = LatentSACAgent(
            intention_time_step_spec,
            intention_action_spec,
            inner_agent=inner_agent,
            model_network=model_network,
            model_optimizer=model_optimizer,
            model_batch_size=model_batch_size,
            num_images_per_summary=num_images_per_summary,
            sequence_length=sequence_length,
            gradient_clipping=gradient_clipping,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter,
            fps=fps
        )
        self.control_agent = LatentSACAgent(
            control_time_step_spec,
            control_action_spec,
            inner_agent=inner_agent,
            model_network=model_network,
            model_optimizer=model_optimizer,
            model_batch_size=model_batch_size,
            num_images_per_summary=num_images_per_summary,
            sequence_length=sequence_length,
            gradient_clipping=gradient_clipping,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter,
            fps=fps
        )

        self.policy = h_latent_actor_policy.HierarchicalLatentActorPolicy(
            intention_time_step_spec=intention_time_step_spec,
            intention_action_spec=intention_action_spec,
            control_time_step_spec=control_time_step_spec,
            control_action_spec=control_action_spec,
            intention_agent=self.intention_agent,
            control_agent=self.control_agent,
            clip=True
        )

        self.collect_policy = h_latent_actor_policy.HierarchicalLatentActorPolicy(
            intention_time_step_spec=intention_time_step_spec,
            intention_action_spec=intention_action_spec,
            control_time_step_spec=control_time_step_spec,
            control_action_spec=control_action_spec,
            intention_agent=self.intention_agent,
            control_agent=self.control_agent,
            clip=False
        )

    def train(self, intention_experience, control_experience, weights=None):
        self.intention_agent.train(intention_experience, weights)
        self.control_agent.train(control_experience, weights)
