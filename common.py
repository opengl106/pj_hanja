import numpy as np
import tensorflow as tf
from tensorflow import keras


def masked_loss(label, logit):
    loss_indexwise = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=keras.losses.Reduction.NONE,
    )
    unmasked = loss_indexwise(label, logit)

    # Mask zeros where length of expected label sequences is variable.
    mask = tf.cast(label != 0, dtype=unmasked.dtype)
    loss = unmasked * mask

    # Probability normalization
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_accuracy(label, logit):
    pred = tf.cast(tf.argmax(logit, axis=-1), dtype=label.dtype)
    unmasked = tf.cast(pred == label, dtype=tf.int32)

    mask = tf.cast(label != 0, dtype=tf.int32)
    hits = unmasked * mask

    return tf.reduce_sum(hits) / tf.reduce_sum(mask)

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
