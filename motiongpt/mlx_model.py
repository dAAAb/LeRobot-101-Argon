"""Native MLX MotionGPT layers: T5 encoder/decoder, cached attention and VQ.

T5 follows Apple's MIT mlx-examples/t5 at commit
796f5b53cab69a3d48a44233ce21aae889e94a08 and HF Transformers 4.44.2.
The VQ layout follows pinned OpenMotionLab/MotionGPT (vendor licenses).
Unlike the minimal example, this preserves padding masks, GELU-new and caches
cross-attention projections, necessary for faithful, efficient FLAN-T5 decoding.
"""
import math

import mlx.core as mx


def relative_buckets(relative, bidirectional):
    buckets = 32
    sign = mx.zeros_like(relative)
    if bidirectional:
        buckets //= 2
        sign = (relative > 0).astype(mx.int32) * buckets
        distance = mx.abs(relative)
    else:
        distance = mx.maximum(-relative, 0)
    exact = buckets // 2
    logarithmic = exact + (mx.log(mx.maximum(distance, 1) / exact) / math.log(128 / exact) * (buckets - exact)).astype(mx.int32)
    return sign + mx.where(distance < exact, distance, mx.minimum(logarithmic, buckets - 1))


class MlxT5:
    def __init__(self, weights):
        self.w = {key.removeprefix('t5.'): value for key, value in weights.items() if key.startswith('t5.')}

    def linear(self, value, name):
        return value @ self.w[name + '.weight'].T

    def norm(self, value, name):
        return mx.fast.rms_norm(value, self.w[name + '.weight'], 1e-6)

    def heads(self, value):
        return value.reshape(value.shape[0], value.shape[1], 12, 64).transpose(0, 2, 1, 3)

    def bias(self, query_length, key_length, stack, offset=0):
        relative = mx.arange(key_length)[None, :] - mx.arange(offset, offset + query_length)[:, None]
        ids = relative_buckets(relative, stack == 'encoder')
        table = self.w[stack + '.block.0.layer.0.SelfAttention.relative_attention_bias.weight']
        return table[ids].transpose(2, 0, 1)[None]

    def attention(self, value, name, mask, cache=None, memory_kv=None):
        query = self.heads(self.linear(value, name + '.q'))
        if memory_kv is not None:
            key, val = memory_kv
        else:
            key = self.heads(self.linear(value, name + '.k'))
            val = self.heads(self.linear(value, name + '.v'))
            if cache is not None:
                key = mx.concatenate((cache[0], key), axis=2)
                val = mx.concatenate((cache[1], val), axis=2)
        # T5 attention deliberately has no 1/sqrt(head_dim) factor.
        attended = mx.fast.scaled_dot_product_attention(query, key, val, scale=1.0, mask=mask)
        merged = attended.transpose(0, 2, 1, 3).reshape(value.shape[0], value.shape[1], 768)
        return self.linear(merged, name + '.o'), (key, val)

    def feed_forward(self, value, name):
        hidden = self.linear(value, name + '.wi_0')
        # FLAN's gated-gelu maps to HF gelu_new (tanh), not exact erf GELU.
        hidden = .5 * hidden * (1 + mx.tanh(math.sqrt(2 / math.pi) * (hidden + .044715 * hidden ** 3)))
        return self.linear(hidden * self.linear(value, name + '.wi_1'), name + '.wo')

    def encode(self, input_ids, attention_mask):
        value = self.w['shared.weight'][input_ids]
        padding = mx.where(attention_mask[:, None, None, :] != 0, 0., -1e9)
        mask = self.bias(value.shape[1], value.shape[1], 'encoder') + padding
        for i in range(12):
            base = f'encoder.block.{i}.layer.'
            update, _ = self.attention(self.norm(value, base + '0.layer_norm'), base + '0.SelfAttention', mask)
            value = value + update
            value = value + self.feed_forward(self.norm(value, base + '1.layer_norm'), base + '1.DenseReluDense')
        return self.norm(value, 'encoder.final_layer_norm')

    def cross_cache(self, memory):
        return [(self.heads(self.linear(memory, f'decoder.block.{i}.layer.1.EncDecAttention.k')),
                 self.heads(self.linear(memory, f'decoder.block.{i}.layer.1.EncDecAttention.v'))) for i in range(12)]

    def decode(self, input_ids, cross_cache, attention_mask, cache=None):
        value = self.w['shared.weight'][input_ids]
        offset = cache[0][0].shape[2] if cache is not None else 0
        cache = cache if cache is not None else [None] * 12
        length = value.shape[1]
        relative = mx.arange(offset + length)[None, :] <= mx.arange(offset, offset + length)[:, None]
        mask = self.bias(length, offset + length, 'decoder', offset) + mx.where(relative, 0., -1e9)
        memory_mask = mx.where(attention_mask[:, None, None, :] != 0, 0., -1e9)
        next_cache = []
        for i in range(12):
            base = f'decoder.block.{i}.layer.'
            update, kv = self.attention(self.norm(value, base + '0.layer_norm'), base + '0.SelfAttention', mask, cache[i])
            value = value + update
            update, _ = self.attention(self.norm(value, base + '1.layer_norm'), base + '1.EncDecAttention', memory_mask, memory_kv=cross_cache[i])
            value = value + update
            value = value + self.feed_forward(self.norm(value, base + '2.layer_norm'), base + '2.DenseReluDense')
            next_cache.append(kv)
        value = self.norm(value, 'decoder.final_layer_norm')
        return self.linear(value, 'lm_head'), next_cache


class MlxVQDecoder:
    def __init__(self, weights):
        self.w = weights

    def conv(self, value, name, dilation=1):
        name = 'vae.decoder.model.' + name
        weight = self.w[name + '.weight']
        return mx.conv1d(value, weight, padding=(weight.shape[1] // 2) * dilation,
                         dilation=dilation) + self.w[name + '.bias']

    def decode(self, codes):
        value = self.w['vae.quantizer.codebook'][codes][None]
        value = mx.maximum(self.conv(value, '0'), 0)
        for block in (2, 3):
            for index, dilation in enumerate((9, 3, 1)):
                name = f'{block}.0.model.{index}'
                residual = self.conv(mx.maximum(value, 0), name + '.conv1', dilation)
                value = value + self.conv(mx.maximum(residual, 0), name + '.conv2')
            value = mx.repeat(value, 2, axis=1)
            value = self.conv(value, f'{block}.2')
        return self.conv(mx.maximum(self.conv(value, '4'), 0), '6')


def inverse_rotate(quaternion, vector):
    axis = -quaternion[..., 1:]
    def cross(a, b):
        return mx.stack((a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                         a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                         a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]), axis=-1)
    uv = cross(axis, vector)
    uuv = cross(axis, uv)
    return vector + 2 * (quaternion[..., :1] * uv + uuv)


def recover_joints(features):
    rotation = features[..., 0]
    angle = mx.cumsum(mx.concatenate((mx.zeros_like(rotation[..., :1]), rotation[..., :-1]), axis=-1), axis=-1)
    zero = mx.zeros_like(angle)
    quaternion = mx.stack((mx.cos(angle), zero, mx.sin(angle), zero), axis=-1)
    xz = mx.concatenate((mx.zeros_like(features[..., :1, 1:3]), features[..., :-1, 1:3]), axis=-2)
    velocity = mx.stack((xz[..., 0], zero, xz[..., 1]), axis=-1)
    integrated = mx.cumsum(inverse_rotate(quaternion, velocity), axis=-2)
    root = mx.stack((integrated[..., 0], features[..., 3], integrated[..., 2]), axis=-1)
    joints = features[..., 4:67].reshape(*features.shape[:-1], 21, 3)
    joints = inverse_rotate(quaternion[..., None, :], joints)
    translation = mx.stack((root[..., 0], zero, root[..., 2]), axis=-1)
    return mx.concatenate((root[..., None, :], joints + translation[..., None, :]), axis=-2)
