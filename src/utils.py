from __future__ import print_function
import numpy as np
from numpy import newaxis
import tensorflow as tf
from sklearn import metrics
import data.dataprocessor as dp
import data.electionprocessor as ep


def load_data(FLAGS, data_source, dynamic_padding=True, preprocessing=False, embedding=True, saved=True, max_length=None):
	if data_source == "lidong":
		with tf.name_scope("input_data"):
			print("Loading Li-Dong data and pre-trained word embedding..")
			data = dp.LidongData(batch_size=FLAGS.batch_size, dynamic_padding=dynamic_padding, preprocessing=preprocessing, 
								embedding=embedding, saved=saved, max_length=max_length)
	elif data_source == "election":
		with tf.name_scope("input_data"):
			print("Loading election data and pre-trained word embedding..")
			data = ep.ElectionData(batch_size=FLAGS.batch_size, dynamic_padding=dynamic_padding, preprocessing=preprocessing, 
								embedding=embedding, saved=saved, max_length=max_length)
	print()
	return data


def embed(df):
	with tf.variable_scope("embed"):
		embedding_init = tf.get_variable("embedding", initializer=tf.constant_initializer(df.glove_vec), shape=df.glove_shape, trainable=False)
		# embedding_init = embedding.assign(df.glove_vec)
	return embedding_init


def fscores(y_test, y_predicted):
	y_test = np.argmax(y_test, 1)
	f1_3class = metrics.f1_score(y_test, y_predicted, average='macro')
	f1_2class = (metrics.f1_score(y_test, y_predicted, average=None)[0]+metrics.f1_score(y_test, y_predicted, average=None)[-1])/2
	return f1_3class, f1_2class


def target_embedding(data, target_index):
	tar_vecs = np.array([])
	for tokens in target_index:
		tar_vec = np.array([])
		for token in tokens:
			token_vec = data.glove_vec[token]
			tar_vec = np.concatenate([tar_vec, token_vec])
		tar_vec = tar_vec.reshape(len(tar_vec)/token_vec.shape[0], token_vec.shape[0])
		tar_vecs = np.concatenate([tar_vecs, tar_vec.mean(axis=0)]) 
	tar_vecs = tar_vecs.reshape((len(target_index), token_vec.shape[0]))
	return tar_vecs[:,newaxis,:]