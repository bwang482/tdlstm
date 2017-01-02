from __future__ import print_function
from __future__ import division
import sys
sys.path.insert(0, r'../')
import time
import argparse
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from data.util import cartesian_product
from utils import load_data
from train_lstm import LSTM
from train_tdlstm import TDLSTM
from train_tclstm import TCLSTM


parser = argparse.ArgumentParser()

# Pick a data set and a LSTM model
parser.add_argument('--data', type=str, default='lidong', help='Choose a data set; lidong or election')
parser.add_argument('--model', type=str, default='LSTM', help='Choose a model; LSTM, TDLSTM or TCLSTM')
parser.add_argument('--tune', action='store_true', help='Whether or not to optimise hyperparameters with grid-search')

# Training parameters
parser.add_argument('--random_state', type=int, default=42, help='Random state initialization for reproducibility')
parser.add_argument('--batch_size', type=int, default=40, help='Mini-batch size')
parser.add_argument('--seq_len', type=int, default=42, help='Sequence length')
parser.add_argument('--num_hidden', type=int, default=200, help='Number of units in the hidden layer')
parser.add_argument('--num_classes', type=int, default=3, help='Number of classes/labels')
# parser.add_argument('--dropout_input', type=float, default=1.0, help='Input keep probability for dropout')
parser.add_argument('--dropout_output', type=float, default=0.8, help='Output keep probability for dropout')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--max_epoch', type=int, default=1000, help='Total number of epochs for training')
parser.add_argument('--early_stopping_rounds', type=int, default=30, help='Number of epochs allowed for setting early stopping criterion')
parser.add_argument('--scoring_metrics', type=str, default='accuracy', help='Classifiaction metrics used for early stopping')

# Session parameters
parser.add_argument('--restore', action='store_true', help='Restore previously trained model')
parser.add_argument('--checkpoint_file', type=str, default='../checkpoints/tclstm_best', help='Checkpoint file path')
parser.add_argument('--allow_soft_placement', type=bool, default=False, help='Allow soft device replacement')
parser.add_argument('--log_device_placement', type=bool, default=False, help='Log placement of ops on devices')

args = parser.parse_args()


def TRAIN(model):
	print("\nParameters:")
	for attr, value in sorted(vars(args).items()):
		print("{}={}".format(attr.upper(), value))
	print()
	print("Graph initialized..")
	t1 = time.time()
	print("time taken:", t1-t0)
	print()
	data = load_data(args, args.data, saved=True)
	run_network(args, data, model, tuning=False)
	
	
def TUNE(model):
	hyperparameters_all = {
		'batch_size': [40],
		'seq_len': [42],
		'num_hidden': [50, 100, 200],
		'learning_rate': [0.001],
		'dropout_output': [0.7, 0.6],
	}
	
	maxx = 0
	data = load_data(args, args.data, saved=True)
	for i, hyperparameters in enumerate(cartesian_product(hyperparameters_all)):
		print("Evaluating hyperparameters:", hyperparameters)
		for attr, value in hyperparameters.items():
			setattr(args, attr, value)
		test_score, eval_score = run_network(args, data, model, tuning=True)
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


def run_network(args, data, model, tuning=False):
	if model == 'LSTM':
		nn = LSTM(args, data, tuning=tuning)
		test_score, eval_score = nn.train_lstm(args, data)
		return test_score, eval_score
	elif model =='TDLSTM':
		nn = TDLSTM(args, data, tuning=tuning)
		test_score, eval_score = nn.train_tdlstm(args, data)
		return test_score, eval_score
	elif model =='TCLSTM':
		nn = TCLSTM(args, data, tuning=tuning)
		test_score, eval_score = nn.train_tclstm(args, data)
		return test_score, eval_score
	else:
		print("No such model; please select from LSTM, TDLSTM or TCLSTM")
			
	
if __name__ == '__main__':
	t0 = time.time()
	if not args.tune:
		TRAIN(args.model)
	else:
		TUNE(args.model)
	# TEST(args.model)
	print()
	print("Total time taken: %f mins"%((time.time()-t0)/60))

