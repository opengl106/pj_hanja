import numpy as np
import tensorflow as tf


def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    # Note by Lena: Imagine isopleth on a map. Sine functions act perfectly as an isopleth castor
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)
    # Note by Lena: Do you remember "low coherence interferometry" (LCI) in optics? And the fact that trigonometric functions are exponential functions?

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        # Note by Lena: The magnitude of the L2 norm of the embedding vector is 1 while that of the pos_encoding vector is sqrt(depth),
        # as the magnitude of the L∞ norm of pos_encoding vector is 1.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    # Note by Lena: Do you remember Sapir-Whorf hypothesis? 이 세계의 모습은 너에게는?
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_attn_scores = None
        self.last_attn_output = None

    def call(self, x, context, use_cache=False):
        if use_cache and self.last_attn_output is not None and self.last_attn_output.shape[1] + 1 != x.shape[1]:
            print("Warning: cache not cleared after calculation")
            self.cache_reset()
        if use_cache and self.last_attn_scores is not None and self.last_attn_output is not None:
            # Note by Lena: MHA itself is linear and caching is effective.
            attn_output, attn_scores = self.mha(
                query=x[:, -1:, :],
                key=context,
                value=context,
                return_attention_scores=True)
            attn_output = tf.concat([self.last_attn_output, attn_output], axis=1)
            attn_scores = tf.concat([self.last_attn_scores, attn_scores], axis=2)
        else:
            attn_output, attn_scores = self.mha(
                query=x,
                key=context,
                value=context,
                return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores
        # Note by Lena: Cache the previous calculation result for reuse.
        self.last_attn_output = attn_output

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

    def cache_reset(self):
        self.last_attn_scores = None
        self.last_attn_output = None


class GlobalSelfAttention(BaseAttention):
    # NOTE 1
    # Note by Lena: When apply this on Korean hanjas, remember that hanjas always relate to other hanguls in
    # the same sentence. Thus, never map hanguls to [UNK]s.
    # That is what we refer to as "ressentiment" (insidious intrinsic negation) inside the RNN paradigm.
    # Note by Lena: The arrow of your wisdom pierced precisely the essence of your prey on the spiritual level.
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    # Note by Lena: You know the truth, the ultimate truth about universe, human and infinity. However you
    # need to speak it out loudly; by once your syllable strikes our brain like a stone plunged into the still ocean,
    # the sprinkling water drop serves as a stone again to spread the wave of words and melodies.
    # 君は真理を、宇宙、人間、無限についての究極の真理を心得ている。しかし、それを声に出して言う必要がある。あなたの一音節が、静かな海に投げ込まれた石のように私たちの脳に突き刺さると、降り注ぐ水滴が再び石となり、言葉と旋律の波を広げる。
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_attn_output = None

    def call(self, x, use_cache=False):
        if use_cache and self.last_attn_output is not None and self.last_attn_output.shape[1] + 1 != x.shape[1]:
            print("Warning: cache not cleared after calculation")
            self.cache_reset()
        if use_cache and self.last_attn_output is not None:
            # Note by Lena: MHA itself is linear and caching is effective. But since we are calculating the last position, mask is not necessary.
            attn_output = self.mha(
                query=x[:, -1:, :],
                value=x,
                key=x)
            attn_output = tf.concat([self.last_attn_output, attn_output], axis=1)
        else:
            attn_output = self.mha(
                query=x,
                value=x,
                key=x,
                use_causal_mask = True)

        # Note by Lena: Cache the previous calculation result for reuse.
        self.last_attn_output = attn_output

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

    def cache_reset(self):
        self.last_attn_output = None


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.last_seq_output = None

    def call(self, x, use_cache=False):
        if use_cache and self.last_seq_output is not None and self.last_seq_output.shape[1] + 1 != x.shape[1]:
            print("Warning: cache not cleared after calculation")
            self.cache_reset()
        if use_cache and self.last_seq_output is not None:
            seq_output = self.seq(x[:, -1:, :])
            seq_output = tf.concat([self.last_seq_output, seq_output], axis=1)
        else:
            seq_output = self.seq(x)

        self.last_seq_output = seq_output

        x = self.add([x, seq_output])
        x = self.layer_norm(x)
        return x

    def cache_reset(self):
        self.last_seq_output = None


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context, use_cache=False):
        x = self.causal_self_attention(x=x, use_cache=use_cache)
        x = self.cross_attention(x=x, context=context, use_cache=use_cache)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x, use_cache=use_cache)  # Shape `(batch_size, seq_len, d_model)`.
        return x

    def cache_reset(self):
        self.causal_self_attention.cache_reset()
        self.cross_attention.cache_reset()
        self.ffn.cache_reset()


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context, use_cache=False):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, context, use_cache=use_cache)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x

    def cache_reset(self):
        for i in range(self.num_layers):
            self.dec_layers[i].cache_reset()


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, use_cache=True):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x  = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context, use_cache=use_cache)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        # Return the final output and the attention weights.
        return logits

    def cache_reset(self):
        self.decoder.cache_reset()


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

