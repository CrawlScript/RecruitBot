# coding=utf-8

import tensorflow as tf
tf.enable_eager_execution()
from recruit_bot.data import get_start_and_end_indices
from tqdm import tqdm

import numpy as np


def predict(recruit_bot, vp, max_len):

    start_index, end_index = get_start_and_end_indices(vp)
    vocab_size = len(vp.vocabulary_)

    prefixes = [[start_index]]
    words = []

    for _ in tqdm(range(max_len)):
        inputs = tf.constant(np.array(prefixes), dtype=tf.int32)#, trainable=False)
        logits = recruit_bot(inputs)[0, -1]
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        while True:
            random_index = np.random.choice(vocab_size, 1, p=probs)[0]
            if random_index != 0:
                break

        prefixes[0].append(random_index)

        if random_index == end_index:
            break
        else:
            words.append(vp.vocabulary_.reverse(random_index))

    return words

