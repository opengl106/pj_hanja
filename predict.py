import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import List

from data_load import load_vocab_list, load_vocab
from hyperparams import Hyperparams as hp
from train import H2HModel

hanguls, hanjas = load_vocab_list()
hangul2idx, idx2hangul, hanja2idx, idx2hanja = load_vocab()

class H2HPredictor():
    """
    A easy-to-import wrapper around the original model.
    """
    def __init__(self) -> None:
        m = H2HModel()
        checkpoint_path = hp.logdir + "/cp.ckpt"
        m.model.load_weights(checkpoint_path)
        self.model = m.model

    def __call__(self, text: str) -> str:
        # call model on single string.
        start = 2
        end = 3
        hangul_indice = [start] + [hangul2idx.get(char, 1) for char in text] + [end] + (hp.maxlen - len(text) - 2) * [0]
        input_tensor = tf.reshape(tf.stack(hangul_indice), [1, hp.maxlen])
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        pbar = tqdm(range(hp.maxlen))
        for i in pbar:
            pbar.set_description(f"Calculating token on {i}th position")
            output = tf.transpose(output_array.stack())[np.newaxis, :]
            predictions = self.model([input_tensor, output], training=False)

            # Select the last token from the `seq_len` dimension.
            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i + 1, predicted_id[0][0])

            if predicted_id[0][0] == end:
                break

        hanja_indice = output[0][1:]
        hanja = "".join([
            (idx2hanja[int(index)] if int(index) != 1 else text[i])
            for i, index in enumerate(hanja_indice)
            if int(hangul_indice[i]) != 0
        ])
        return hanja

    def convert(self, article: List[str]) -> List[str]:
        # predict on article consist of multiple strings.
        start = 2
        end = 3
        hangul_indices = [
            [start] + [hangul2idx.get(char, 1) for char in line] + [end] + (hp.maxlen - len(line) - 2) * [0]
            for line in article
        ]
        input_tensor = tf.convert_to_tensor(
            [tf.stack(indice) for indice in hangul_indices]
        )
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        bulk_size = input_tensor.shape[0]
        end_sentences = tf.cast(np.ones([bulk_size, 1]), dtype=tf.int64)
        ends = tf.cast(np.array(end)[np.newaxis, np.newaxis], dtype=tf.int64)
        for j in range(bulk_size):
            output_array = output_array.write(0 * bulk_size + j, start)

        pbar = tqdm(range(hp.maxlen))
        for i in pbar:
            pbar.set_description(f"Calculating tokens on {i}th position")
            output = tf.transpose(tf.reshape(tf.transpose(output_array.stack()), [i + 1, bulk_size]))
            predictions = self.model([input_tensor, output], training=False)

            # Select the last token from the `seq_len` dimension.
            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            end_sentences *= tf.cast(predicted_id != ends, tf.int64)
            predicted_id *= end_sentences
            for j in range(bulk_size):
                output_array = output_array.write((i + 1) * bulk_size + j, predicted_id[j][0])

            if tf.reduce_sum(end_sentences) == 0:
                break

        hanja_indices = output[:, 1:]
        return ["".join([idx2hanja[item] for item in line]) for line in hanja_indices.numpy()]
