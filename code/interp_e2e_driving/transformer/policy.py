import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from transformer import StableTransformerXL


class TransformerGaussianPolicy():
    def __init__(self, state_dim, act_dim, n_transformer_layers=4, n_attn_heads=3):
        """
            NOTE - I/P Shape : [seq_len, batch_size, state_dim]
        """
        super(TransformerGaussianPolicy, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim

        self.transformer = StableTransformerXL(d_input=state_dim, n_layers=n_transformer_layers,
                                               n_heads=n_attn_heads, d_head_inner=32, d_ff_inner=64)
        self.memory = None

        self.head_sate_value = tf.keras.layers.Dense(1, input_shape=(state_dim, ))
        self.head_act_mean = tf.keras.layers.Dense(act_dim, input_shape=(state_dim, ))
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = tf.Variable(tf.constant(log_std, dtype=tf.float32))

        self.relu = tf.keras.layers.ReLU()

    def _distribution(self, trans_state):
        mean = tf.tanh(self.head_act_mean(trans_state))
        std = tf.exp(self.log_std)

        return tfp.distributions.Normal(mean, std)

    def _log_prob_from_distribution(self, policy, action):
        return policy.log_prob(action).sum(axis=-1)

    def __call__(self, state, action=None):
        trans_state = self.transformer(state, self.memory)
        trans_state, self.memory = trans_state['logits'], trans_state['memory']

        policy = self._distribution(trans_state)
        state_value = self.head_sate_value(trans_state)

        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(policy, action)

        return policy, logp_a, state_value

    def step(self, state):
        if state.shape[0] == self.state_dim:
            state = state.reshape(1, 1, -1)

        trans_state = self.transformer(state, self.memory)
        trans_state, self.memory = trans_state['logits'], trans_state['memory']

        policy = self._distribution(trans_state)
        action = policy.sample()
        logp_a = self._log_prob_from_distribution(policy, action)
        state_value = self.head_sate_value(trans_state)

        return action.numpy(), logp_a.numpy(), state_value.numpy()


if __name__ == '__main__':
    states = np.random.randn(20, 5, 8)  # seq_size, batch_size, dim - better if dim % 2 == 0
    print("=> Testing Policy")
    policy = TransformerGaussianPolicy(state_dim=states.shape[-1], act_dim=4)
    for i in range(10):
        act = policy(states)
        action = act[0].sample()
        print(tf.reduce_any(tf.math.is_nan(action)), action.shape)
