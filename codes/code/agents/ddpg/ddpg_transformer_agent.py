# This file is modified from <https://github.com/tensorflow/agents>

"""A DDPG Agent with modification to handle multiple source inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.policies import ou_noise_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils

from code.policies import actor_transformer_policy


class DdpgInfo(collections.namedtuple(
    'DdpgInfo', ('actor_loss', 'critic_loss'))):
    pass


@gin.configurable
class TransformerDdpgAgent(tf_agent.TFAgent):
    """A DDPG Agent."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 actor_network,
                 critic_network,
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

        tf.Module.__init__(self, name=name)
        self._actor_network = actor_network
        actor_network.create_variables()
        if target_actor_network:
            target_actor_network.create_variables()
        self._target_actor_network = common.maybe_copy_target_network_with_checks(
            self._actor_network, target_actor_network, 'TargetActorNetwork')
        self._critic_network = critic_network
        critic_network.create_variables()
        if target_critic_network:
            target_critic_network.create_variables()
        self._target_critic_network = common.maybe_copy_target_network_with_checks(
            self._critic_network, target_critic_network, 'TargetCriticNetwork')

        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer

        self._ou_stddev = ou_stddev
        self._ou_damping = ou_damping
        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._dqda_clipping = dqda_clipping
        self._td_errors_loss_fn = (
                td_errors_loss_fn or common.element_wise_huber_loss)
        self._gamma = gamma
        self._reward_scale_factor = reward_scale_factor
        self._gradient_clipping = gradient_clipping

        self._update_target = self._get_target_updater(
            target_update_tau, target_update_period)

        policy = actor_transformer_policy.ActorPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            actor_network=self._actor_network, clip=True)
        collect_policy = actor_transformer_policy.ActorPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            actor_network=self._actor_network, clip=False)
        collect_policy = ou_noise_policy.OUNoisePolicy(
            collect_policy,
            ou_stddev=self._ou_stddev,
            ou_damping=self._ou_damping,
            clip=True)

        super(TransformerDdpgAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy,
            collect_policy,
            train_sequence_length=2 if not self._actor_network.state_spec else None,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)

    def _initialize(self):
        common.soft_variables_update(
            self._critic_network.variables,
            self._target_critic_network.variables,
            tau=1.0)
        common.soft_variables_update(
            self._actor_network.variables,
            self._target_actor_network.variables,
            tau=1.0)

    def _get_target_updater(self, tau=1.0, period=1):
        """Performs a soft update of the target network parameters.

    For each weight w_s in the original network, and its corresponding
    weight w_t in the target network, a soft update is:
    w_t = (1- tau) x w_t + tau x ws

    Args:
      tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
      period: Step interval at which the target networks are updated.
    Returns:
      An operation that performs a soft update of the target network parameters.
    """
        with tf.name_scope('get_target_updater'):
            def update():
                """Update target network."""
                # TODO(b/124381161): What about observation normalizer variables?
                critic_update = common.soft_variables_update(
                    self._critic_network.variables,
                    self._target_critic_network.variables,
                    tau,
                    tau_non_trainable=1.0)
                actor_update = common.soft_variables_update(
                    self._actor_network.variables,
                    self._target_actor_network.variables,
                    tau,
                    tau_non_trainable=1.0)
                return tf.group(critic_update, actor_update)

            return common.Periodically(update, period, 'periodic_update_targets')

    def _experience_to_transitions(self, experience):
        transitions = trajectory.to_transition(experience)

        # Remove time dim if we are not using a recurrent network.
        if not self._actor_network.state_spec:
            transitions = tf.nest.map_structure(lambda x: tf.squeeze(x, [1]),
                                                transitions)

        time_steps, policy_steps, next_time_steps = transitions
        actions = policy_steps.action
        return time_steps, actions, next_time_steps

    def _train(self, experience, weights=None):
        time_steps, actions, next_time_steps = self._experience_to_transitions(
            experience)

        # TODO(b/124382524): Apply a loss mask or filter boundary transitions.
        trainable_critic_variables = self._critic_network.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_critic_variables, ('No trainable critic variables to '
                                                'optimize.')
            tape.watch(trainable_critic_variables)
            critic_loss = self.critic_loss(time_steps, actions, next_time_steps,
                                           weights=weights)
        tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
        critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
        self._apply_gradients(critic_grads, trainable_critic_variables,
                              self._critic_optimizer)

        trainable_actor_variables = self._actor_network.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_actor_variables, ('No trainable actor variables to '
                                               'optimize.')
            tape.watch(trainable_actor_variables)
            actor_loss = self.actor_loss(time_steps, weights=weights)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
        self._apply_gradients(actor_grads, trainable_actor_variables,
                              self._actor_optimizer)

        self.train_step_counter.assign_add(1)
        self._update_target()

        # TODO(b/124382360): Compute per element TD loss and return in loss_info.
        total_loss = actor_loss + critic_loss
        return tf_agent.LossInfo(total_loss,
                                 DdpgInfo(actor_loss, critic_loss))

    def _apply_gradients(self, gradients, variables, optimizer):
        # Tuple is used for py3, where zip is a generator producing values once.
        grads_and_vars = tuple(zip(gradients, variables))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                             self._gradient_clipping)

        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(grads_and_vars,
                                                self.train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
                                                self.train_step_counter)

        optimizer.apply_gradients(grads_and_vars)

    def critic_loss(self,
                    time_steps,
                    actions,
                    next_time_steps,
                    weights=None):
        """Computes the critic loss for DDPG training.

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      next_time_steps: A batch of next timesteps.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.
    Returns:
      critic_loss: A scalar critic loss.
    """
        with tf.name_scope('critic_loss'):
            target_actions, _ = self._target_actor_network(
                next_time_steps.observation, next_time_steps.step_type)
            target_critic_net_input = (next_time_steps.observation, target_actions)
            target_q_values, _ = self._target_critic_network(
                target_critic_net_input, next_time_steps.step_type)

            td_targets = tf.stop_gradient(
                self._reward_scale_factor * next_time_steps.reward +
                self._gamma * next_time_steps.discount * target_q_values)

            critic_net_input = (time_steps.observation, actions)
            q_values, _ = self._critic_network(critic_net_input,
                                               time_steps.step_type)

            critic_loss = self._td_errors_loss_fn(td_targets, q_values)
            if nest_utils.is_batched_nested_tensors(
                    time_steps, self.time_step_spec, num_outer_dims=2):
                # Do a sum over the time dimension.
                critic_loss = tf.reduce_sum(critic_loss, axis=1)
            if weights is not None:
                critic_loss *= weights
            critic_loss = tf.reduce_mean(critic_loss)

            with tf.name_scope('Losses/'):
                tf.compat.v2.summary.scalar(
                    name='critic_loss', data=critic_loss, step=self.train_step_counter)

            if self._debug_summaries:
                td_errors = td_targets - q_values
                common.generate_tensor_summaries('td_errors', td_errors,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('td_targets', td_targets,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('q_values', q_values,
                                                 self.train_step_counter)

            return critic_loss

    def actor_loss(self, time_steps, weights=None):
        """Computes the actor_loss for DDPG training.

    Args:
      time_steps: A batch of timesteps.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.
      # TODO(b/124383618): Add an action norm regularizer.
    Returns:
      actor_loss: A scalar actor loss.
    """
        with tf.name_scope('actor_loss'):
            actions, _ = self._actor_network(time_steps.observation,
                                             time_steps.step_type)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(actions)
                q_values, _ = self._critic_network((time_steps.observation, actions),
                                                   time_steps.step_type)
                actions = tf.nest.flatten(actions)

            dqdas = tape.gradient([q_values], actions)

            actor_losses = []
            for dqda, action in zip(dqdas, actions):
                if self._dqda_clipping is not None:
                    dqda = tf.clip_by_value(dqda, -1 * self._dqda_clipping,
                                            self._dqda_clipping)
                loss = common.element_wise_squared_loss(
                    tf.stop_gradient(dqda + action), action)
                if nest_utils.is_batched_nested_tensors(
                        time_steps, self.time_step_spec, num_outer_dims=2):
                    # Sum over the time dimension.
                    loss = tf.reduce_sum(loss, axis=1)
                if weights is not None:
                    loss *= weights
                loss = tf.reduce_mean(loss)
                actor_losses.append(loss)

            actor_loss = tf.add_n(actor_losses)

            with tf.name_scope('Losses/'):
                tf.compat.v2.summary.scalar(
                    name='actor_loss', data=actor_loss, step=self.train_step_counter)

        return actor_loss
