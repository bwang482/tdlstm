import sys
sys.path.insert(0, r'/home/ccrmad/Code/TDLSTM')
import data.util
import tensorflow as tf
import numpy as np


class TDLSTMClassifier:

	def __init__(self, args, embedding_init):
		self.learning_rate = args.learning_rate
		self.num_hidden = args.num_hidden
		self.num_classes = args.num_classes
		self.embedding_init = embedding_init
		self.xl = tf.placeholder(tf.int32, [None, None])
		self.xr = tf.placeholder(tf.int32, [None, None])
		self.y = tf.placeholder(tf.int32, [None, self.num_classes])
		self.seq_len_l = tf.placeholder(tf.int64, [None])
		self.seq_len_r = tf.placeholder(tf.int64, [None])


	def inference(self, reuse=None):

		embed_inputs_fw = tf.nn.embedding_lookup(self.embedding_init, self.xl) ## (batch_size, seq_len, 100)
		embed_inputs_bw = tf.nn.embedding_lookup(self.embedding_init, self.xr) ## (batch_size, seq_len, 100)

		with tf.variable_scope('hidden', reuse=reuse):
			with tf.variable_scope('forward_lstm_cell'):
				lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_hidden, use_peepholes=False, 
													# forget_bias=0.0, 
													# activation=tf.nn.relu, 
													# initializer=tf.truncated_normal_initializer(stddev=0.1),
													# initializer=tf.contrib.layers.xavier_initializer(),
													initializer=tf.random_uniform_initializer(-0.003, 0.003),
													state_is_tuple=True)
				if not reuse:
					lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=0.5, output_keep_prob=0.7)
				# lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_fw_cell] * 4, state_is_tuple=True)

			with tf.variable_scope('backward_lstm_cell'):
				lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_hidden, use_peepholes=False, 
													# forget_bias=0.0, 
													# activation=tf.nn.relu,
													# initializer=tf.truncated_normal_initializer(stddev=0.1),
													# initializer=tf.contrib.layers.xavier_initializer(),
													initializer=tf.random_uniform_initializer(-0.003, 0.003),
													state_is_tuple=True)
				if not reuse:
					lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=0.5, output_keep_prob=0.7)
				# lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_bw_cell] * 4, state_is_tuple=True)

			# fw_initial_state = lstm_fw_cell.zero_state(self.batch_size, tf.float32)
			# bw_initial_state = lstm_bw_cell.zero_state(self.batch_size, tf.float32)

			with tf.variable_scope("forward_lstm"):
				outputs_fw, output_states_fw  = tf.nn.dynamic_rnn(
					cell=lstm_fw_cell,
					inputs=embed_inputs_fw,
					dtype=tf.float32,
					sequence_length=self.seq_len_l
					)       ## (batch_size, seq_len_l, num_hidden)

			with tf.variable_scope("backward_lstm"):
				outputs_bw, output_states_bw  = tf.nn.dynamic_rnn(
					cell=lstm_bw_cell,
					inputs=embed_inputs_bw,
					dtype=tf.float32,
					sequence_length=self.seq_len_r
					)       ## (batch_size, seq_len_r, num_hidden)

			last_outputs_fw = self.last_relevant(outputs_fw, self.seq_len_l) ## (batch_size, num_hidden)
			last_outputs_bw = self.last_relevant(outputs_bw, self.seq_len_r) ## (batch_size, num_hidden)
			last_outputs = tf.concat(1, [last_outputs_fw, last_outputs_bw]) ## (batch_size, num_hidden*2)

		with tf.variable_scope('output', reuse=reuse):
			with tf.variable_scope('softmax'):
				W = tf.get_variable('W', [self.num_hidden*2, self.num_classes],
									initializer=tf.random_uniform_initializer(-0.003, 0.003))
									# initializer=tf.contrib.layers.xavier_initializer())
									# initializer=tf.truncated_normal_initializer(stddev=0.1))
				b = tf.get_variable('b', [self.num_classes], initializer=tf.constant_initializer(0.1))
			logits = tf.matmul(last_outputs, W) + b
		return logits


	def loss(self, logits):
		cost = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(self.y, tf.float32))
		mean_cost = tf.reduce_mean(cost)
		y_pred = tf.argmax(logits, 1)
		correct_pred = tf.equal(y_pred, tf.argmax(self.y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		return mean_cost, accuracy, y_pred

	
	def training(self, cost):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		# optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
		train_op = optimizer.minimize(cost)
		return train_op


	@staticmethod
	def seq_length(data):
		used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
		length = tf.reduce_sum(used, reduction_indices=1)
		length = tf.cast(length, tf.int64)
		return length

	@staticmethod
	def last_relevant(outputs, length):
		# Borrowed from: https://gist.github.com/rockt/f4f9df5674f3da6a32786bcf9fbb6a88
		batch_size, max_length, hidden_size = tf.unpack(tf.shape(outputs))
		index = tf.range(0, batch_size) * max_length + (tf.cast(length, tf.int32) - 1)
		flat = tf.reshape(outputs, [-1, hidden_size])
		relevant = tf.gather(flat, index)
		return relevant
