from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn import metrics
import data.dataprocessor as dp
import data.electionprocessor as ep


def load_data(FLAGS, data_source, dynamic_padding=True, preprocessing=False, embedding=True, saved=True, max_length=None):
	if data_source is "lidong":
		with tf.device("/cpu:0"), tf.name_scope("input_data"):
			print("Loading Li-Dong data and pre-trained word embedding..")
			data = dp.LidongData(batch_size=FLAGS.batch_size, dynamic_padding=dynamic_padding, preprocessing=preprocessing, 
								embedding=embedding, saved=saved, max_length=max_length)
	elif data_source is "election":
		with tf.device("/cpu:0"), tf.name_scope("input_data"):
			print("Loading election data and pre-trained word embedding..")
			data = ep.ElectionData(batch_size=FLAGS.batch_size, dynamic_padding=dynamic_padding, preprocessing=preprocessing, 
								embedding=embedding, saved=saved, max_length=max_length)
	print()
	return data


def embed(df):
	with tf.device("/cpu:0"), tf.variable_scope("embed"):
		embedding = tf.get_variable("embedding", shape=df.glove_shape, trainable=False)
		embedding_init = embedding.assign(df.glove_vec)
	return embedding_init


def fscores(y_test, y_predicted):
	y_test = np.argmax(y_test, 1)
	f1_3class = metrics.f1_score(y_test, y_predicted, average='macro')
	f1_2class = (metrics.f1_score(y_test, y_predicted, average=None)[0]+metrics.f1_score(y_test, y_predicted, average=None)[-1])/2
	return f1_3class, f1_2class