from __future__ import division
import gensim
import numpy as np
import itertools
from itertools import product
from collections import Counter 


def gensim_load_vec(path):
    gensim_emb = gensim.models.Word2Vec.load_word2vec_format(path, binary=False)
    vocab = gensim_emb.index2word
    vec = gensim_emb.syn0
    shape = gensim_emb.syn0.shape
    return gensim_emb, vec, shape, vocab


def build_vocabulary(sequences, add_pad_token=None):
    """
    Builds a vocabulary mapping from token to index for all tokens in the sequences.
    Returns the vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sequences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    if add_pad_token is not None:
        vocabulary_inv += [add_pad_token]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def pad_sequences(sequences, dynamic_padding=False, pad_location="LEFT", max_length=None):
    """
    Pads all sequences to the same length. The length is defined by the longest sequence.
    Returns padded sequences.
    """
    if dynamic_padding:
        pad_token=1193514
    else:
        pad_token="[PAD]"

    if not max_length:
        max_length = max(len(x) for x in sequences)

    result = []
    for i in range(len(sequences)):
        sentence = sequences[i]
        num_padding = max_length - len(sentence)
        if num_padding == 0:
            new_sentence = sentence
        elif num_padding < 0:
            new_sentence = sentence[:num_padding]
        elif pad_location == "RIGHT":
            new_sentence = sentence + [pad_token] * num_padding
        elif pad_location == "LEFT":
            new_sentence = [pad_token] * num_padding + sentence
        else:
            raise Error("Invalid pad_location. Specify LEFT or RIGHT.")
        result.append(new_sentence)
    return result


def batch_iter(data, batch_size, num_epochs, seed=None, fill=False):
    """
    Generates a batch iterator for a dataset.
    """
    random = np.random.RandomState(seed)
    data = np.array(data)
    data_length = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)
    if len(data) % batch_size != 0:
        num_batches_per_epoch += 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = random.permutation(np.arange(data_length))
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_length)
            selected_indices = shuffle_indices[start_index:end_index]
            # If we don't have enough data left for a whole batch, fill it randomly
            if fill is True and end_index >= data_length:
                num_missing = batch_size - len(selected_indices)
                selected_indices = np.concatenate([selected_indices, random.randint(0, data_length, num_missing)])
            yield data[selected_indices]


def xy_iter(x, y, batch_size, num_epochs, random_seed=42, fill=True):
    train_iter = batch_iter(list(zip(x, y)), batch_size, num_epochs, fill=fill, seed=random_seed)
    return map(lambda batch: zip(*batch), train_iter)


def cartesian_product(dicts):
    return [dict(zip(dicts,x)) for x in product(*dicts.values())]

