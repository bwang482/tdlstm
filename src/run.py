from __future__ import print_function
from __future__ import division
import sys
sys.path.insert(0, r'../')
import time
import argparse
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from train_lstm import LSTM
from data.util import cartesian_product
from utils import load_data


parser = argparse.ArgumentParser()

# Pick a data set
parser.add_argument('--data', type=str, default='election', help='Choose a data set; lidong or election')

# Training parameters
parser.add_argument('--random_state', type=int, default=42, help='Random state initialization for reproducibility')
parser.add_argument('--batch_size', type=int, default=50, help='Mini-batch size')
parser.add_argument('--seq_len', type=int, default=42, help='Sequence length')
parser.add_argument('--num_hidden', type=int, default=100, help='Number of units in the hidden layer')
parser.add_argument('--num_classes', type=int, default=3, help='Number of classes/labels')
parser.add_argument('--dropout_input', type=float, default=1.0, help='Input keep probability for dropout')
parser.add_argument('--dropout_output', type=float, default=1.0, help='Output keep probability for dropout')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--max_epoch', type=int, default=1000, help='Total number of epochs for training')
parser.add_argument('--early_stopping_rounds', type=int, default=50, help='Number of epochs allowed for setting early stopping criterion')
parser.add_argument('--scoring_metrics', type=str, default='accuracy', help='Classifiaction metrics used for early stopping')

# Session parameters
parser.add_argument('--restore', type=bool, default=False, help='Restore previously trained model')
parser.add_argument('--checkpoint_file', type=str, default='../checkpoints/lstm_best', help='Checkpoint file path')
parser.add_argument('--allow_soft_placement', type=bool, default=False, help='Allow soft device replacement')
parser.add_argument('--log_device_placement', type=bool, default=False, help='Log placement of ops on devices')

args = parser.parse_args()

def TRAIN():
	print("\nParameters:")
	for attr, value in sorted(vars(args).items()):
		print("{}={}".format(attr.upper(), value))
	print()
	print("Graph initialized..")
	t1 = time.time()
	print("time taken:", t1-t0)
	print()
	data = load_data(args, args.data, saved=True)
	nn = LSTM(args, data, tuning=False)
	nn.train_lstm(args, data)
	
	
def TUNE():
	hyperparameters_all = {
		'batch_size': [40, 50, 60],
		'seq_len': [42],
		'num_hidden': [50, 100, 200],
		'learning_rate': [0.001],
		'dropout_output': [0.7, 0.8],
	}
	
	maxx = 0
	data = load_data(args, args.data, saved=False)
	for i, hyperparameters in enumerate(cartesian_product(hyperparameters_all)):
		print("Evaluating hyperparameters:", hyperparameters)
		for attr, value in hyperparameters.items():
			setattr(args, attr, value)
		nn = LSTM(args, data, tuning=True)
		test_score, eval_score = nn.train_lstm(args, data)
		if eval_score[0] > maxx:
			maxx = eval_score[0]
			best_score = test_score
			hyperparameters_best = hyperparameters
		tf.reset_default_graph()
	print()
	print("Optimisation finished..")
	print("Optimised hyperparameters:")
	for attr, value in sorted(hyperparameters_best.items()):
		print("{}={}".format(attr.upper(), value))
	print()
	print("Final Test Data Accuracy = {:.5f}; 3-class F1 = {:.5f}; 2-class F1 = {:.5f}"
		                  .format(best_score[0], best_score[1], best_score[2]))
			
	
if __name__ == '__main__':
	t0 = time.time()
	TUNE()
	# TEST()
	print()
	print("Total time taken: %f mins"%((time.time()-t0)/60))

