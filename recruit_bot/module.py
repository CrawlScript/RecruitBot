# coding=utf-8

import tensorflow as tf
import math

from recruit_bot.data import get_start_and_end_indices
from tqdm import tqdm
tf.enable_eager_execution()

from tensorflow.python import keras
import numpy as np


class RecruitBot(keras.Model):
    def __init__(self, vocab_size, embedding_size, drop_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.partial_embeddings = tf.Variable(
            tf.truncated_normal([vocab_size - 1, embedding_size], stddev=1 / math.sqrt(embedding_size)))
        self.lstm = keras.layers.LSTM(embedding_size, return_sequences=True)
        self.dropout_layer = keras.layers.Dropout(drop_rate)
        self.dense_layer = keras.layers.Dense(vocab_size)

    def call(self, inputs, training=None, mask=None):
        embeddings = tf.concat([
            tf.zeros([1, self.embedding_size], dtype=tf.float32),
            self.partial_embeddings
        ], axis=0)

        embedded = tf.nn.embedding_lookup(embeddings, inputs)

        lstm_output = self.lstm(embedded, initial_state=self.create_initial_state(inputs))
        dropped_lstm_output = self.dropout_layer(lstm_output, training=training)
        logits = self.dense_layer(dropped_lstm_output)
        return logits

    def create_initial_state(self, inputs):
        if isinstance(inputs, list):
            batch_size = len(inputs)
        else:
            batch_size = inputs.shape[0]
        states = [
            tf.zeros([batch_size, self.embedding_size], dtype=tf.float32),
            tf.zeros([batch_size, self.embedding_size], dtype=tf.float32)
        ]
        return states

    def predict_words(self, vp, max_len):
        vocab_size = len(vp.vocabulary_)
        start_index, end_index = get_start_and_end_indices(vp)

        current_inputs = [start_index]

        embeddings = tf.concat([
            tf.zeros([1, self.embedding_size], dtype=tf.float32),
            self.partial_embeddings
        ], axis=0)

        h, c = self.create_initial_state(current_inputs)
        outputs_list = []

        for i in tqdm(range(max_len - 1)):
            current_inputs = tf.constant(current_inputs, dtype=tf.int32)
            embedded = tf.nn.embedding_lookup(embeddings, current_inputs)
            _, [h, c] = self.lstm.cell(embedded, [h, c])
            logits = self.dense_layer(h)

            probs = tf.nn.softmax(logits).numpy()
            outputs = []
            for prob in probs:
                for _ in range(5):
                    random_index = np.random.choice(vocab_size, 1, p=prob)[0]
                    if random_index != 0:
                        break
                outputs.append(random_index)
            outputs_list.append(outputs)
            current_inputs = outputs

        indices_list = np.stack(outputs_list, axis=1)
        words_list = []
        for indices in indices_list:
            words = []
            for index in indices:
                if index in [0, start_index]:
                    continue
                if index == end_index:
                    break
                else:
                    words.append(vp.vocabulary_.reverse(index))
            words_list.append(words)
        return words_list

