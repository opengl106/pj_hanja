# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Training.
'''

import codecs
import os

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from common import masked_accuracy, masked_loss
from data_load import load_vocab, load_data
from models import CustomSchedule, Transformer
from hyperparams import Hyperparams as hp


# Load vocab
hangul2idx, idx2hangul, hanja2idx, idx2hanja = load_vocab()


class H2HModel():
    """
    A tensorflow 2 transformer version of the original model.
    """
    def __init__(self, model=None):
        if model:
            self.model = model
            return None

        self.model = Transformer(
            num_layers=hp.num_layers,
            d_model=hp.d_model,
            num_heads=hp.num_heads,
            dff=hp.dff,
            input_vocab_size=len(hangul2idx),
            target_vocab_size=len(hanja2idx),
            dropout_rate=hp.dropout_rate)

        learning_rate = CustomSchedule(hp.d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)

        self.model.compile(
            optimizer=optimizer,
            loss=masked_loss,
            metrics=[masked_accuracy],
        )

        return None

    def predict(self, data, dtype=tf.int32, batch_size=hp.batch_size):
        logit = self.model.predict(data, batch_size=batch_size)
        pred = tf.cast(tf.argmax(logit, axis=-1), dtype=dtype)
        return pred


if __name__ == '__main__':
    # Data loading
    X_train, Y_train, L_train = load_data(mode="train")
    train_data_x = tf.data.Dataset.from_tensor_slices(X_train)
    train_data_y = tf.data.Dataset.from_tensor_slices(Y_train)
    train_data_l = tf.data.Dataset.from_tensor_slices(L_train)
    train_batches = tf.data.Dataset.zip((tf.data.Dataset.zip((train_data_x, train_data_y)), train_data_l)).shuffle(hp.buffer_size).batch(hp.batch_size)

    X_val, Y_val, L_val = load_data(mode="val")
    val_data_x = tf.data.Dataset.from_tensor_slices(X_val)
    val_data_y = tf.data.Dataset.from_tensor_slices(Y_val)
    val_data_l = tf.data.Dataset.from_tensor_slices(L_val)
    val_batches = tf.data.Dataset.zip((tf.data.Dataset.zip((val_data_x, val_data_y)), val_data_l)).shuffle(hp.buffer_size).batch(hp.batch_size)

    # Model loading
    if os.path.exists(hp.logdir + "/model.keras"):
        print(f"Loading dataset from {hp.logdir}/model.keras")
        model = keras.models.load_model(hp.logdir + "/model.keras", custom_objects={
            'masked_accuracy': masked_accuracy,
            'masked_loss': masked_loss,
        })
        keras.backend.set_value(model.optimizer.learning_rate, hp.learning_rate)
        m = H2HModel(model)
    else:
        m = H2HModel()

    # Training
    checkpoint_path = hp.logdir + "/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )
    history = m.model.fit(
        train_batches,
        epochs=hp.num_epochs,
        validation_data=val_batches,
        callbacks=[cp_callback],
    )

    if not os.path.exists(hp.logdir):
        os.mkdir(hp.logdir)
    m.model.save(hp.logdir + "/model.keras")

    # Logging
    hist_df = pd.DataFrame(history.history)
    hist_json_file = hp.logdir + "/history.json"
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    hist_csv_file = hp.logdir + "/history.csv"
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    # Evaluating
    preds = m.predict(val_batches[0])
    pred_data = tf.data.Dataset.from_tensor_slices(preds)
    with codecs.open(hp.logdir + "/eval.txt", 'w', 'utf-8') as fout:
        for xx, yy, pred in tf.data.Dataset.zip(val_data_x, val_data_l, pred_data): # sentence-wise
            inputs, expected, got = [], [], []
            for xxx, yyy, ppp in zip(xx, yy, pred):  # character-wise
                if int(xxx)==0: break
                inputs.append(idx2hangul[int(xxx)])
                expected.append(idx2hanja[int(yyy)] if int(yyy)!=1 else idx2hangul[int(xxx)])
                got.append(idx2hanja[int(ppp)] if int(ppp) != 1 else idx2hangul[int(xxx)])
            fout.write(u"* Input   : {}\n".format("".join(inputs)))
            fout.write(u"* Expected: {}\n".format("".join(expected)))
            fout.write(u"* Got     : {}\n".format("".join(got)))
            fout.write("\n")
