import sys
sys.path.insert(0, r'../')
import data.util
import tensorflow as tf
import numpy as np


class LSTMClassifier:

	def __init__(self, args, embedding_init):
		self.learning_rate = args.learning_rate
		self.num_hidden = args.num_hidden
		self.num_classes = args.num_classes
		self.dropout_output = args.dropout_output
		self.dropout_input = args.dropout_input
		self.clip_norm = args.clip_norm

		self.embedding_init = embedding_init
		self.x = tf.placeholder(tf.int32, [None, None], 'input')
		self.y = tf.placeholder(tf.int32, [None, self.num_classes], 'labels')
		self.seq_len = tf.placeholder(tf.int64, [None], 'input_length')


	def inference(self, forward_only=None):

		embed_inputs = tf.nn.embedding_lookup(self.embedding_init, self.x) ## (batch_size, seq_len, 100)

		with tf.variable_scope('hidden', reuse=forward_only):
			with tf.variable_scope('lstm_cell'):
				lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_hidden, use_peepholes=False, 
													# forget_bias=0.0, 
													activation=tf.nn.relu, 
													# initializer=tf.truncated_normal_initializer(stddev=0.1),
													# initializer=tf.random_uniform_initializer(-0.003, 0.003),
													initializer=tf.contrib.layers.xavier_initializer(),
													state_is_tuple=True)
				if not forward_only:
					lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.dropout_output)
				# lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell] * 4, state_is_tuple=True)

			if not forward_only:
				embed_inputs = tf.nn.dropout(embed_inputs, keep_prob=self.dropout_input)

			rnn_outputs, output_states  = tf.nn.dynamic_rnn(
				cell=lstm_cell,
				inputs=embed_inputs,
				dtype=tf.float32,
				sequence_length=self.seq_len,
				)       ## (batch_size, seq_len, num_hidden)

			# rnn_outputs = tf.transpose(rnn_outputs, perm=[1,0,2]) ## (seq_len, batch_size, num_hidden) NOT NEEDED ANY MORE
			last_outputs = self.last_relevant(rnn_outputs, self.seq_len) ## (batch_size, num_hidden)

		with tf.variable_scope('output', reuse=forward_only):
			with tf.variable_scope('softmax'):
				W = tf.get_variable('W', [self.num_hidden, self.num_classes],
									# initializer=tf.random_uniform_initializer(-0.003, 0.003))
									initializer=tf.contrib.layers.xavier_initializer())
									# initializer=tf.truncated_normal_initializer(stddev=0.1))
				b = tf.get_variable('b', [self.num_classes], initializer=tf.constant_initializer(0.1))
			logits = tf.matmul(last_outputs, W) + b
		self.embed_inputs = embed_inputs
		return logits


	def loss(self, logits, forward_only=None):
		cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(self.y, tf.float32))
		mean_cost = tf.reduce_mean(cost)
		y_pred = tf.argmax(logits, 1)
		correct_pred = tf.equal(y_pred, tf.argmax(self.y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		if forward_only:
			str_summary_type = 'eval'
			loss_summ = tf.summary.scalar("{0}_loss".format(str_summary_type), mean_cost)
			acc_summ = tf.summary.scalar("{0}_accuracy".format(str_summary_type), accuracy)
			merged = tf.summary.merge([loss_summ, acc_summ])
			return mean_cost, accuracy, y_pred, merged
		else:
			return mean_cost, accuracy, y_pred

	
	def training(self, cost):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		# train_op = optimizer.minimize(cost)
		
		trainables = tf.trainable_variables()
		grads = tf.gradients(cost, trainables)
		grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.clip_norm)
		capped_gvs = zip(grads, trainables)
		train_op = optimizer.apply_gradients(capped_gvs)
		return train_op


	@staticmethod
	def seq_length(data):
		used = tf.sign(tf.reduce_max(tf.abs(data), axis=2))
		length = tf.reduce_sum(used, axis=1)
		length = tf.cast(length, tf.int64)
		return length

	@staticmethod
	def last_relevant(outputs, length):
		# Borrowed from: https://gist.github.com/rockt/f4f9df5674f3da6a32786bcf9fbb6a88
		batch_size, max_length, hidden_size = tf.unstack(tf.shape(outputs))
		index = tf.range(0, batch_size) * max_length + (tf.cast(length, tf.int32) - 1)
		flat = tf.reshape(outputs, [-1, hidden_size])
		relevant = tf.gather(flat, index)
		return relevant
