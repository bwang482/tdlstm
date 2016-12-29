from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import time
import os
import numpy as np
from sklearn import metrics
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
from data import util
from models.tdlstm_classifier import TDLSTMClassifier
import data.dataprocessor as dp
import data.electionprocessor as ep
from laplotter import LossAccPlotter


# Pick a data set
tf.flags.DEFINE_string("data", "lidong", "Choose a data set")

# Training parameters
tf.flags.DEFINE_integer("random_state", 42, "Random state initialization for reproducibility")
tf.flags.DEFINE_integer("batch_size", 64, "Mini-batch size")
tf.flags.DEFINE_integer("seq_len", 42, "Sequence length")
tf.flags.DEFINE_integer("num_hidden", 256, "Number of units in the hidden layer")
tf.flags.DEFINE_integer("num_classes", 3, "Number of classes/labels")
tf.flags.DEFINE_integer("learning_rate", 0.0001, "Learning rate for the optimizer")
# tf.flags.DEFINE_integer("max_epoch", 5, "Number of epochs trained with the initial learning rate")
tf.flags.DEFINE_integer("max_max_epoch", 1600, "Total number of epochs for training")
# tf.flags.DEFINE_integer("lr_decay", 0.05, "The decay of learning rate for each epoch after 'max_epoch' ")
tf.flags.DEFINE_integer("early_stopping_rounds", 150, "Number of epochs allowed for setting early stopping criterion")
tf.flags.DEFINE_string("scoring_metrics", 'accuracy', "Classifiaction metrics used for early stopping")

# Session parameters
tf.flags.DEFINE_boolean("restore", False, "Restore previously trained model")
tf.flags.DEFINE_string("checkpoint_file", 'checkpoints/tdlstm_best', "Checkpoint file path")
tf.flags.DEFINE_boolean("allow_soft_placement", False, "Allow soft device replacement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def load_data(data_source):
	if data_source is "lidong":
		with tf.device("/cpu:0"), tf.name_scope("input_data"):
			print("Loading Li-Dong data and pre-trained word embedding..")
			data = dp.LidongData(batch_size=FLAGS.batch_size, dynamic_padding=True, preprocessing=False, 
								embedding=True, saved=True, max_length=None)
	elif data_source is "election":
		with tf.device("/cpu:0"), tf.name_scope("input_data"):
			print("Loading election data and pre-trained word embedding..")
			data = ep.ElectionData(batch_size=FLAGS.batch_size, dynamic_padding=True, preprocessing=False, 
								embedding=True, saved=True, max_length=None)
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


def test(session, early_stopping_rounds, early_stopping_metric_list, early_stopping_metric_minimize=False, metrics=FLAGS.scoring_metrics):
	feed = {model.xl: data.dev_left_x, model.xr: data.dev_right_x, model.y: data.dev_y, 
			model.seq_len_l: data.dev_left_size, model.seq_len_r: data.dev_right_size}
	test_loss_value, acc_test, pred = session.run(test_loss, feed)
	f1_3class, f1_2class = fscores(data.dev_y, pred)
	print("*** Validation Loss = {:.6f}; Validation Accuracy = {:.5f}; 3-class F1 = {:.5f}; 2-class F1 = {:.5f}"
				.format(test_loss_value, acc_test, f1_3class, f1_2class))
	print()
	early_stop = False
	early_stopping_score = -1
	if metrics == 'accuracy':
		early_stopping_score = acc_test
		early_stopping_metric_list.append(acc_test)
	elif metrics == '3classf1':
		early_stopping_score = f1_3class
		early_stopping_metric_list.append(f1_3class)
	elif metrics == '2classf1':
		early_stopping_score = f1_2class
		early_stopping_metric_list.append(f1_2class)
	assert early_stopping_score > 0

	if (not FLAGS.restore) and (early_stopping_metric_minimize):
		if all(early_stopping_score <= i for i in early_stopping_metric_list):
			saver.save(sess, FLAGS.checkpoint_file)
		if early_stopping_metric_list[::-1].index(min(early_stopping_metric_list)) > early_stopping_rounds:
			early_stop = True
		return (test_loss_value, early_stopping_score, early_stop)
	elif (not FLAGS.restore) and (not early_stopping_metric_minimize):
		if all(early_stopping_score >= i for i in early_stopping_metric_list):
			saver.save(sess, FLAGS.checkpoint_file)
		if early_stopping_metric_list[::-1].index(max(early_stopping_metric_list)) > early_stopping_rounds:
			early_stop = True
		return (test_loss_value, early_stopping_score, early_stop)


def final_test(session, feed, labels):
	saver.restore(session, FLAGS.checkpoint_file)
	test_loss_value, acc_test, pred = session.run(test_loss, feed)
	f1_3class, f1_2class = fscores(labels, pred)
	print("****** Final test Loss = {:.6f}; Test Accuracy = {:.5f}; 3-class F1 = {:.5f}; 2-class F1 = {:.5f}"
				.format(test_loss_value, acc_test, f1_3class, f1_2class))
	print()


data = load_data(FLAGS.data)
embedding_init = embed(data)

model = TDLSTMClassifier(FLAGS, embedding_init)
logits = model.inference()
train_loss = model.loss(logits)
train_op = model.training(train_loss[0])

pred = model.inference(reuse=True)
test_loss = model.loss(pred)

scoring_list = []
# Visualizing loss function and accuracy during training over epochs
plotter = LossAccPlotter(title="Training plots",
                         save_to_filepath="img/tdlstm_plot.png",
                         show_regressions=False,
                         show_averages=False,
                         show_loss_plot=True,
                         show_acc_plot=True,
                         show_plot_window=True,
                         x_label="Epoch")

init = tf.group(tf.initialize_all_variables(),
				tf.initialize_local_variables())


with tf.Session() as sess:
	t0 = time.time()
	saver = tf.train.Saver()
	if FLAGS.restore and FLAGS.checkpoint_file:
		print()
		print("Loading variables from '%s' .." % FLAGS.checkpoint_file)
		print()
		saver.restore(sess, FLAGS.checkpoint_file)
		feed = {model.xl: data.test_left_x, model.xr: data.test_right_x, model.y: data.test_y, 
				model.seq_len_l: data.test_left_size, model.seq_len_r: data.test_right_size}
		final_test(sess, feed, data.test_y)
	else:
		sess.run(init)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		print("\nParameters:")
		for attr, value in sorted(FLAGS.__flags.items()):
			print("{}={}".format(attr.upper(), value))
		print()
		print("Graph initialized..")
		t1 = time.time()
		print("time taken:", t1-t0)
		print()

		np.random.seed(FLAGS.random_state)
		for epoch in range(FLAGS.max_max_epoch):
			data.reset_batch_pointer()

			total_loss = 0.0
			total_acc = 0.0
			# fw_state, bw_state = sess.run([model.fw_initial_state, model.bw_initial_state])
			for step in range(data.num_batches):
				x, y, seq_length, xl, seq_length_l, xr, seq_length_r = data.next_batch()
				feed={model.xl: xl, model.xr: xr, model.y: y, model.seq_len_l: seq_length_l, model.seq_len_r: seq_length_r}
				sess.run(train_op, feed)
				current_loss, current_acc, _ = sess.run(train_loss, feed)
				total_loss += current_loss
				total_acc += current_acc

			# feed={model.xl: xl, model.xr: xr, model.y: y, model.seq_len_l: seq_length_l, model.seq_len_r: seq_length_r}
			# train_loss_value, acc_train, _ = sess.run(test_loss, feed)
			print()
			print("Epoch {:2d}: Training loss = {:.6f}; Training Accuracy = {:.5f}".format(epoch+1, total_loss/data.num_batches, total_acc/data.num_batches))

			dev_loss_value, dev_score, early_stop = test(sess, FLAGS.early_stopping_rounds, scoring_list)
			plotter.add_values(epoch,
							loss_train=total_loss/data.num_batches, acc_train=total_acc/data.num_batches,
							loss_val=dev_loss_value, acc_val=dev_score)
			if early_stop:
				print('Early stopping...')
				print('DONE')
				t2 = time.time()
				print("time taken: %f mins"%((t2-t1)/60))
				break

		feed = {model.xl: data.test_left_x, model.xr: data.test_right_x, model.y: data.test_y, 
				model.seq_len_l: data.test_left_size, model.seq_len_r: data.test_right_size}
		final_test(sess, feed, data.test_y)

		coord.request_stop()
		coord.join(threads)

if not FLAGS.restore:
	plotter.block()

