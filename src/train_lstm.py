from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import time
import os
import numpy as np
from operator import itemgetter
from sklearn import metrics
import tensorflow as tf
from models.lstm_classifier import LSTMClassifier
from utils import *
from laplotter import LossAccPlotter


class LSTM:
	def __init__(self, args, data, tuning):
		self.FLAGS = args
		self.data = data
		self.tuning = tuning
		self.embedding_init = embed(self.data)

		self.model = LSTMClassifier(self.FLAGS, self.embedding_init)
		logits = self.model.inference()
		self.train_loss = self.model.loss(logits)
		self.train_op = self.model.training(self.train_loss[0])

		pred = self.model.inference(forward_only=True)
		self.test_loss = self.model.loss(pred, forward_only=True)

		# Visualizing loss function and accuracy during training over epochs
		self.plotter = LossAccPlotter(title="Training plots",
		                     save_to_filepath="../img/lstm_plot.png",
		                     show_regressions=False,
		                     show_averages=False,
		                     show_loss_plot=True,
		                     show_acc_plot=True,
		                     show_plot_window=True,
		                     x_label="Epoch")

		self.init = tf.group(tf.initialize_all_variables(),
					tf.initialize_local_variables())
		print("Network initialized..")


	def train_lstm(self, FLAGS, data):
		model = self.model
		scoring_list = []
		best_eval_score = []

		with tf.Session(config=tf.ConfigProto(
			allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)) as sess:
			t0 = time.time()
			saver = tf.train.Saver()
			writer = tf.train.SummaryWriter("../output")
			if FLAGS.restore and FLAGS.checkpoint_file:
				print()
				print("Loading model from '%s' .." % FLAGS.checkpoint_file)
				print()
				saver.restore(sess, FLAGS.checkpoint_file)
				# feed = {model.x: data.dev_x, model.y: data.dev_y, model.seq_len: data.dev_size}
				feed = {model.x: data.test_x, model.y: data.test_y, model.seq_len: data.test_size}
				test_loss_value, test_score = self.final_test(sess, feed)
			else:
				sess.run(self.init)
				coord = tf.train.Coordinator()
				threads = tf.train.start_queue_runners(sess=sess, coord=coord)

				np.random.seed(FLAGS.random_state)
				for epoch in range(FLAGS.max_epoch):
					data.reset_batch_pointer()

					total_loss = 0.0
					total_acc = 0.0
					for step in range(data.num_batches):
						x, y, seq_length, _, _, _, _, _ = data.next_batch()
						feed={model.x: x, model.y: y, model.seq_len: seq_length}
						sess.run(self.train_op, feed)
						current_loss, current_acc, _ = sess.run(self.train_loss, feed)
						total_loss += current_loss
						total_acc += current_acc
					
					if not self.tuning:
						print()
						print("Epoch {:2d}: Training loss = {:.6f}; Training Accuracy = {:.5f}".format(epoch+1, total_loss/data.num_batches, total_acc/data.num_batches))

					feed = {model.x: data.dev_x, model.y: data.dev_y, model.seq_len: data.dev_size}
					dev_loss_value, dev_score, early_stop, eval_summary = self.eval(sess, feed, saver, FLAGS.early_stopping_rounds, \
						                                         scoring_list, False, FLAGS.scoring_metrics)
					best_eval_score.append(dev_score)
					# str_summary_type = 'train'
					# loss_summ = tf.scalar_summary("{0}_loss".format(str_summary_type), self.total_train_loss.assign(total_loss/data.num_batches))
					# acc_summ = tf.scalar_summary("{0}_accuracy".format(str_summary_type), self.total_train_acc.assign(total_acc/data.num_batches))
					# train_summary = tf.merge_summary([loss_summ, acc_summ])
					# writer.add_summary(train_summary, epoch)
					# writer.add_summary(eval_summary, epoch)
					
					self.plotter.add_values(epoch,
									loss_train=total_loss/data.num_batches, acc_train=total_acc/data.num_batches,
									loss_val=dev_loss_value, acc_val=dev_score[0])

					if early_stop:
						print('Early stopping after %s epoches...' % str(epoch))
						best_eval_score = max(best_eval_score,key=itemgetter(1)) if FLAGS.scoring_metrics=='3classf1' else \
						                  max(best_eval_score,key=itemgetter(0)) if FLAGS.scoring_metrics=='accuracy' \
						                  else max(best_eval_score,key=itemgetter(2))
						print("Final dev loss = {:.5f}; Dev Accuracy = {:.5f}; 3-class F1 = {:.5f}; 2-class F1 = {:.5f}"
								.format(dev_loss_value, best_eval_score[0], best_eval_score[1], best_eval_score[2]))
						if not self.tuning:
							t1 = time.time()
							print("time taken: %f mins"%((t1-t0)/60))
						break

				writer.close()
				feed = {model.x: data.test_x, model.y: data.test_y, model.seq_len: data.test_size}
				test_loss_value, test_score = self.final_test(sess, feed)

				coord.request_stop()
				coord.join(threads)

		# if not FLAGS.restore:
		# 	self.plotter.block()

		return test_score, best_eval_score


	def eval(self, session, feed, saver, early_stopping_rounds, early_stopping_metric_list, early_stopping_metric_minimize=False, metrics='accuracy'):
		test_loss_value, acc_test, pred, eval_summary = session.run(self.test_loss, feed)
		f1_3class, f1_2class = fscores(self.data.dev_y, pred)
		if not self.tuning:
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

		if (not self.FLAGS.restore) and (early_stopping_metric_minimize): # For minimising the eval score
			if all(early_stopping_score <= i for i in early_stopping_metric_list):
				saver.save(session, self.FLAGS.checkpoint_file)
				best_eval_score = (acc_test, f1_3class, f1_2class)
			if early_stopping_metric_list[::-1].index(min(early_stopping_metric_list)) > early_stopping_rounds:
				early_stop = True
			return (test_loss_value, (acc_test, f1_3class, f1_2class), early_stop)
		elif not (self.FLAGS.restore and early_stopping_metric_minimize):  # For maximising the eval score
			if all(early_stopping_score >= i for i in early_stopping_metric_list):
				saver.save(session, self.FLAGS.checkpoint_file)
				best_eval_score = (acc_test, f1_3class, f1_2class)
			if early_stopping_metric_list[::-1].index(max(early_stopping_metric_list)) > early_stopping_rounds:
				early_stop = True
			return (test_loss_value, (acc_test, f1_3class, f1_2class), early_stop, eval_summary)


	def final_test(self, session, feed):
		tf.train.Saver().restore(session, self.FLAGS.checkpoint_file)
		test_loss_value, acc_test, pred, _ = session.run(self.test_loss, feed)
		true_labels = self.data.test_y
		f1_3class, f1_2class = fscores(true_labels, pred)
		print("****** Final test Loss = {:.6f}; Test Accuracy = {:.5f}; 3-class F1 = {:.5f}; 2-class F1 = {:.5f}"
					.format(test_loss_value, acc_test, f1_3class, f1_2class))
		print()
		return (test_loss_value, (acc_test, f1_3class, f1_2class))


