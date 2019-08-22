# coding=utf-8
import tensorflow as tf

from recruit_bot.data import build_vocab, preprocess_all, get_start_and_end_indices
from recruit_bot.inference import predict

tf.enable_eager_execution()
import tensorflow.contrib.learn as learn
import os
from recruit_bot.module import RecruitBot
import numpy as np
from tqdm import tqdm

training = False

if training:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

max_len = 50


embedding_size = 150
drop_rate = 0.3

batch_size = 50
vocab_path = "vocab.p"
model_path = "model/model"
model_dir = os.path.dirname(model_path)
checkpoint_path = tf.train.latest_checkpoint(model_dir)


if training:
    with open("title.txt", "r", encoding="utf-8") as f:
        text_list = [line.strip() for line in f.readlines()]
        words_list = preprocess_all(text_list)

    if checkpoint_path is None:
        vp = build_vocab(words_list, max_len, vocab_path)
    else:
        vp = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    indices_list = np.array(list(vp.transform(words_list)), dtype=np.int32)
else:
    vp = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

vocab_size = len(vp.vocabulary_)
recuit_bot = RecruitBot(vocab_size, embedding_size, drop_rate)

start_index, end_index = get_start_and_end_indices(vp)

optimizer = tf.train.AdamOptimizer(learning_rate=5e-3)
checkpoint = tf.train.Checkpoint(
    optimizer=optimizer,
    model=recuit_bot,
    global_step=tf.train.get_or_create_global_step()
)

if training:

    if checkpoint_path is not None:
        print("restore")
        checkpoint.restore(checkpoint_path)

    for epoch in range(1000):
        for step, batch_data in tqdm(enumerate(tf.data.Dataset.from_tensor_slices(indices_list).shuffle(1000).batch(batch_size))):
            batch_input = batch_data[:, :-1]
            batch_output = batch_data[:, 1:]

            with tf.GradientTape() as tape:
                logits = recuit_bot(batch_input, training=True)
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=tf.one_hot(batch_output, depth=vocab_size)
                )

            vars = tape.watched_variables()
            grads = tape.gradient(losses, vars)
            optimizer.apply_gradients(zip(grads, vars), global_step=tf.train.get_or_create_global_step())

            if step == 0 and epoch % 5 == 0:
                words = recuit_bot.predict_words(vp, max_len)[0]
                print("".join(words))

                mean_loss = tf.reduce_mean(losses)
                checkpoint.save(model_path)
                print(epoch, step, mean_loss)
else:
    checkpoint.restore(tf.train.latest_checkpoint(model_dir))
    words = recuit_bot.predict_words(vp, max_len)[0]
    print("".join(words))

    from flask import Flask

    app = Flask('aop')


    @app.route("/")
    def index():
        words_list = recuit_bot.predict_words(vp, max_len)
        texts = ["".join(words) for words in words_list]
        return "\n".join(texts)


    app.run(host='0.0.0.0', port=5002)




