import tensorflow as tf
import numpy as np


def compute_triu(x, diagonal):
    x = x.numpy()
    y = np.triu(x, diagonal)
    return tf.convert_to_tensor(y)


class PositionalEmbedding(tf.Module):
    def __init__(self, dim):
        super(PositionalEmbedding, self).__init__()

        self.dim = dim
        inv_freq = 1 / (10000 ** (tf.range(0.0, dim, 2.0) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        sinusoid_inp = tf.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = tf.concat([sinusoid_inp.sin(), sinusoid_inp.cos()], axis=-1)
        return pos_emb[:, None, :]


class PositionwiseFF(tf.Module):
    def __init__(self, d_input, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.dropout = dropout
        self.ff = tf.keras.Sequential(
            tf.keras.layers.Dense(d_input, d_inner),
            tf.keras.layers.ReLU(inplace=True),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_inner, d_input),
            tf.keras.layers.Dropout(dropout),
        )

    def forward(self, input_):
        ff_out = self.ff(input_)
        return ff_out


class GatingMechanism(tf.Module):
    def __init__(self, d_input, bg=0.1):
        super(GatingMechanism, self).__init__()
        self.Wr = tf.keras.layers.Dense(d_input, d_input)
        self.Ur = tf.keras.layers.Dense(d_input, d_input)
        self.Wz = tf.keras.layers.Dense(d_input, d_input)
        self.Uz = tf.keras.layers.Dense(d_input, d_input)
        self.Wg = tf.keras.layers.Dense(d_input, d_input)
        self.Ug = tf.keras.layers.Dense(d_input, d_input)
        self.bg = bg

        self.sigmoid = tf.sigmoid()
        self.tanh = tf.tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(tf.multiply(r, x)))
        g = tf.multiply(1 - z, x) + tf.multiply(z, h)
        return g


class MultiHeadAttentionXL(tf.Module):
    def __init__(self, d_input, d_inner, n_heads=4, dropout=0.1, dropouta=0.0):
        super(MultiHeadAttentionXL, self).__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.n_heads = n_heads

        # Linear transformation for keys & values for all heads at once for efficiency.
        # 2 for keys & values.
        self.linear_kv = tf.keras.layers.Dense(d_input, (d_inner * n_heads * 2), use_bias=False)
        # for queries (will not be concatenated with memorized states so separate).
        self.linear_q = tf.keras.layers.Dense(d_input, d_inner * n_heads, use_bias=False)

        # for positional embeddings.
        self.linear_p = tf.keras.layers.Dense(d_input, d_inner * n_heads, use_bias=False)
        self.scale = 1 / (d_inner ** 0.5)  # for scaled dot product attention
        self.dropa = tf.keras.layers.Dropout(dropouta)

        self.lout = tf.keras.layers.Dense(d_inner * n_heads, d_input, use_bias=False)
        self.dropo = tf.keras.layers.Dropout(dropout)

    def _rel_shift(self, x):
        # x shape: [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        with tf.device(x.device):
            zero_pad = tf.zeros(
                (x.size(0), 1, *x.size()[2:]), dtype=x.dtype
            )
            concated = tf.concat([zero_pad, x], axis=1)
            reshaped = tf.reshape(concated, (x.size(1) + 1, x.size(0), *x.size()[2:]))[1:]
            result = tf.reshape(reshaped, x.size())
        return result

    def forward(self, input_, pos_embs, memory, u, v, mask=None):
        """
        + pos_embs: positional embeddings passed separately to handle relative positions.
        + Arguments
            - input: torch.FloatTensor, shape - (seq, bs, self.d_input) = (20, 5, 8)
            - pos_embs: torch.FloatTensor, shape - (seq + prev_seq, bs, self.d_input) = (40, 1, 8)
            - memory: torch.FloatTensor, shape - (prev_seq, b, d_in) = (20, 5, 8)
            - u: torch.FloatTensor, shape - (num_heads, inner_dim) = (3 x )
            - v: torch.FloatTensor, shape - (num_heads, inner_dim)
            - mask: torch.FloatTensor, Optional = (20, 40, 1)
        + Returns
            - output: torch.FloatTensor, shape - (seq, bs, self.d_input)
        + symbols representing shape of the tensors
            - cs: current sequence length, b: batch, H: no. of heads
            - d: inner dimension, ps: previous sequence length
        """
        cur_seq = input_.shape[0]
        prev_seq = memory.shape[0]
        H, d = self.n_heads, self.d_inner
        # concat memory across sequence dimension
        # input_with_memory = [seq + prev_seq x B x d_input] = [40 x 5 x 8]
        input_with_memory = tf.concat([memory, input_], axis=0)

        # k_tfmd, v_tfmd = [seq + prev_seq x B x n_heads.d_head_inner], [seq + prev_seq x B x n_heads.d_head_inner]
        k_tfmd, v_tfmd = tf.split(
            self.linear_kv(input_with_memory),
            2,
            axis=-1,
        )
        # q_tfmd = [seq x B x n_heads.d_head_inner] = [20 x 5 x 96]
        q_tfmd = self.linear_q(input_)

        _, bs, _ = q_tfmd.shape
        assert bs == k_tfmd.shape[1]

        # content_attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        content_attn = tf.einsum(
            "ibhd,jbhd->ijbh",
            (
                (q_tfmd.view(cur_seq, bs, H, d) + u),
                k_tfmd.view(cur_seq + prev_seq, bs, H, d),
            ),
        )

        # p_tfmd: [seq + prev_seq x 1 x n_heads.d_head_inner] = [40 x 1 x 96]
        p_tfmd = self.linear_p(pos_embs)
        # position_attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        position_attn = tf.einsum(
            "ibhd,jhd->ijbh",
            (
                (q_tfmd.view(cur_seq, bs, H, d) + v),
                p_tfmd.view(cur_seq + prev_seq, H, d),
            ),
        )

        position_attn = self._rel_shift(position_attn)
        # attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        attn = content_attn + position_attn

        if mask is not None and mask.any().item():
            # fills float('-inf') where mask is True.
            attn = attn.masked_fill(mask[..., None], -float("inf"))
        # rescale to prevent values from exploding.
        # normalize across the value sequence dimension.
        attn = tf.nn.softmax(attn * self.scale, axis=1)
        # attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        attn = self.dropa(attn)

        # attn_weighted_values = [curr x B x n_heads.d_inner] = [20 x 5 x 96]
        attn_weighted_values = (
            tf.einsum(
                "ijbh,jbhd->ibhd",
                (
                    attn,  # (cs, cs + ps, b, H)
                    v_tfmd.view(cur_seq + prev_seq, bs, H, d),  # (cs + ps, b, H, d)
                ),
            )  # (cs, b, H, d)
            .contiguous()  # we need to change the memory layout to make `view` work
            .view(cur_seq, bs, H * d)
        )  # (cs, b, H * d)

        # output = [curr x B x d_input] = [20 x 5 x 8]
        output = self.dropo(self.lout(attn_weighted_values))
        return output


class StableTransformerEncoderLayerXL(tf.Module):
    def __init__(
        self,
        n_heads,
        d_input,
        d_head_inner,
        d_ff_inner,
        dropout,
        gating=True,
        dropouta=0.0,
    ):
        super(StableTransformerEncoderLayerXL, self).__init__()

        self.gating = gating
        self.gate1 = GatingMechanism(d_input)
        self.gate2 = GatingMechanism(d_input)
        self.mha = MultiHeadAttentionXL(
            d_input,
            d_head_inner,
            n_heads=n_heads,
            dropout=dropout,
            dropouta=dropouta,
        )
        self.ff = PositionwiseFF(d_input, d_ff_inner, dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(d_input)
        self.norm2 = tf.keras.layers.LayerNormalization(d_input)

    def forward(self, input_, pos_embs, u, v, mask=None, mems=None):
        src2 = self.norm1(input_)
        src2 = self.mha(src2, pos_embs, mems, u, v, mask=mask)
        src = self.gate1(input_, src2) if self.gating else input_ + src2
        src2 = self.ff(self.norm2(src))
        src = self.gate2(src, src2) if self.gating else src + src2
        return src


class StableTransformerXL(tf.Module):
    def __init__(
        self,
        d_input,
        n_layers,
        n_heads,
        d_head_inner,
        d_ff_inner,
        dropout=0.1,
        dropouta=0.0,
    ):
        super(StableTransformerXL, self).__init__()

        (
            self.n_layers,
            self.n_heads,
            self.d_input,
            self.d_head_inner,
            self.d_ff_inner,
        ) = (n_layers, n_heads, d_input, d_head_inner, d_ff_inner)

        self.pos_embs = PositionalEmbedding(d_input)
        self.drop = tf.keras.layers.Dropout(dropout)
        self.layers = [
                StableTransformerEncoderLayerXL(
                    n_heads,
                    d_input,
                    d_head_inner=d_head_inner,
                    d_ff_inner=d_ff_inner,
                    dropout=dropout,
                    dropouta=dropouta,
                )
                for _ in range(n_layers)
            ]

        # u and v are global parameters: maybe changing these to per-head parameters might help performance?
        self.u, self.v = (
            # [n_heads x d_head_inner] = [3 x 32]
            tf.Variable(tf.Tensor(self.n_heads, self.d_head_inner, dtype=tf.float32)),
            tf.Variable(tf.Tensor(self.n_heads, self.d_head_inner, dtype=tf.float32)),
        )

    def init_memory(self, device=tf.device('/cpu:0')):
        with device:
            return [
                # torch.empty(0, dtype=torch.float).to(device)
                tf.zeros(20, 5, 8, dtype=tf.float32)
                for _ in range(self.n_layers + 1)
            ]

    def update_memory(self, previous_memory, hidden_states):
        """
        + Arguments
            - previous_memory: List[torch.FloatTensor],
            - hidden_states: List[torch.FloatTensor]
        """
        assert len(hidden_states) == len(previous_memory)
        mem_len, seq_len = previous_memory[0].size(0), hidden_states[0].size(0)
        # mem_len, seq_len = 3, hidden_states[0].size(0)
        # print(mem_len, seq_len)

        new_memory = []
        end_idx = mem_len + seq_len
        beg_idx = max(0, end_idx - mem_len)
        for m, h in zip(previous_memory, hidden_states):
            cat = tf.concat([m, h], axis=0)
            new_memory.append(cat[beg_idx:end_idx].detach())
        return new_memory

    def forward(self, inputs, memory=None):
        """
        + Arguments
            - inputs - torch.FloatTensor = [T x B x d_inner] = [20 x 5 x 8]
            - memory - Optional, list[torch.FloatTensor] = [[T x B x d_inner] x 5]
        """
        if memory is None:
            memory = self.init_memory(inputs.device)
        assert len(memory) == len(self.layers) + 1

        cur_seq, bs = inputs.shape[:2]
        prev_seq = memory[0].size(0)

        # dec_attn_mask = [curr x curr + prev x 1] = [20 x 40 x 1]
        dec_attn_mask = (
            compute_triu(
                tf.ones((cur_seq, cur_seq + prev_seq)),
                diagonal= prev_seq,
            )
            .bool()[..., None]
            .to(inputs.device)
        )

        pos_ips = tf.range(cur_seq + prev_seq - 1, -1, -1.0, dtype=tf.float32).to(
            inputs.device
        )
        # pos_embs = [curr + prev x 1 x d_input] = [40 x 1 x 8]
        pos_embs = self.drop(self.pos_embs(pos_ips))
        if self.d_input % 2 != 0:
            pos_embs = pos_embs[:, :, :-1]

        hidden_states = [inputs]
        layer_out = inputs
        for mem, layer in zip(memory, self.layers):
            # layer_out = [curr x B x d_inner] = [20 x 5 x 8]
            layer_out = layer(
                layer_out,
                pos_embs,
                self.u,
                self.v,
                mask=dec_attn_mask,
                mems=mem,
            )
            hidden_states.append(layer_out)

        # Memory is treated as a const., don't propagate through it
        # new_memory = [[T x B x d_inner] x 4]
        memory = self.update_memory(memory, hidden_states)
        return {"logits": layer_out, "memory": memory}