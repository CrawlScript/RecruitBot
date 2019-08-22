# coding=utf-8
from tensorflow.contrib import learn
import numpy as np


def preprocess(text):
    return ["<S>"] + list(text) + ["<E>"]


def preprocess_all(text_list):
    return [preprocess(text) for text in text_list]


def tokenizer_fn(s):
    return s


def build_vocab(words_list, max_len, vocab_path):
    vp = learn.preprocessing.VocabularyProcessor(max_document_length=max_len, tokenizer_fn=tokenizer_fn, min_frequency=10)
    vp.fit(words_list)
    vp.save(vocab_path)
    return vp


def get_start_and_end_indices(vp):
    return vp.vocabulary_.get("<S>"), vp.vocabulary_.get("<E>")
