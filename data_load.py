#/usr/bin/python3
# coding: utf-8

import numpy as np
import codecs
from hyperparams import Hyperparams as hp

def load_vocab_list():
    hanguls, hanjas = set(), set()
    for line in codecs.open('data/bible_ko.tsv', 'r', 'utf-8'):
        hangul_sent, hanja_sent = line.strip().split("\t")
        for hangul, hanja in zip(hangul_sent, hanja_sent):
            hanguls.add(hangul)
            hanjas.add(hanja)
    hanjas = hanjas - hanguls
    hanguls = sorted(list(hanguls))
    # NOTE 1
    # Note by Lena: As now we are using the transformer paradigm, the self-attention between the input
    # hanjas and the input hanguls in the same sentence must be taken into consideration.
    hanjas = hanguls + sorted(list(hanjas))
    return hanguls, hanjas

def load_vocab():
    hanguls, hanjas = load_vocab_list()
    hanguls = ['', '[UNK]', '[START]', '[END]'] + hanguls
    hanjas = ['', '[UNK]', '[START]', '[END]'] + hanjas

    hangul2idx = {hangul: idx for idx, hangul in enumerate(hanguls)}
    idx2hangul = {idx: hangul for idx, hangul in enumerate(hanguls)}

    hanja2idx = {hanja: idx for idx, hanja in enumerate(hanjas)}
    idx2hanja = {idx: hanja for idx, hanja in enumerate(hanjas)}

    return hangul2idx, idx2hangul, hanja2idx, idx2hanja

def load_data(mode="train"):
    hangul2idx, idx2hangul, hanja2idx, idx2hanja = load_vocab()

    # Vectorize
    xs, ys, ls = [], [], []  # vectorized sentences
    for line in codecs.open('data/bible_ko.tsv', 'r', 'utf-8'):
        hangul_sent, hanja_sent = line.strip().split("\t")
        if len(hangul_sent) <= hp.maxlen - 2:
            x = [2] + [hangul2idx.get(hangul, 1) for hangul in hangul_sent] + [3]
            y = [2] + [hanja2idx.get(hanja, 1) for hanja in hanja_sent]
            l = [hanja2idx.get(hanja, 1) for hanja in hanja_sent] + [3]

            x.extend([0] * (hp.maxlen - len(x)))  # zero post-padding
            y.extend([0] * (hp.maxlen - len(y)))  # zero post-padding
            l.extend([0] * (hp.maxlen - len(l)))  # zero post-padding

            xs.append(x)
            ys.append(y)
            ls.append(l)

    # Convert to 2d-arrays
    X = np.array(xs, np.int32)
    Y = np.array(ys, np.int32)
    L = np.array(ls, np.int32)

    if mode=="train":
        X, Y, L = X[:-hp.batch_size], Y[:-hp.batch_size], L[:-hp.batch_size]
    else: # eval
        X, Y, L = X[-hp.batch_size:], Y[-hp.batch_size:], L[-hp.batch_size:]

    return X, Y, L