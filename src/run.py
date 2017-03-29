from __future__ import print_function
from __future__ import division
import os, sys
sys.path.insert(0, r'../')
import time
import argparse
from optimise import TRAIN, TUNE, hyperoptTUNE, skoptTUNE


parser = argparse.ArgumentParser()

# Pick a data set and a LSTM model
parser.add_argument('--data', type=str, default='lidong', help='Choose a data set; lidong or election')
parser.add_argument('--load_data', action='store_true', help='Load previously saved data')
parser.add_argument('--model', type=str, default='LSTM', help='Choose a model; LSTM, TDLSTM or TCLSTM')
parser.add_argument('--tune', action='store_true', help='Whether or not to optimise hyperparameters')
parser.add_argument('--tuning_method', type=str, default='skopt', help='Which optimization method to use: grid, rand, hyperopt or skopt')
parser.add_argument('--num_calls', type=int, default=10, help='Number of settings sampled for hyper-parameter tuning')

# Training parameters
parser.add_argument('--random_state', type=int, default=42, help='Random state initialization for reproducibility')
parser.add_argument('--batch_size', type=int, default=51, help='Mini-batch size')
parser.add_argument('--seq_len', type=int, default=42, help='Sequence length')
parser.add_argument('--num_hidden', type=int, default=382, help='Number of units in the hidden layer')
parser.add_argument('--num_classes', type=int, default=3, help='Number of classes/labels')
parser.add_argument('--dropout_input', type=float, default=0.4, help='Input keep probability for dropout')
parser.add_argument('--dropout_output', type=float, default=0.4, help='Output keep probability for dropout')
parser.add_argument('--clip_norm', type=float, default=0.5, help='Gradient clipping ratio')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')

parser.add_argument('--max_epoch', type=int, default=1000, help='Total number of epochs for training')
parser.add_argument('--early_stopping_rounds', type=int, default=20, help='Number of epochs allowed for setting early stopping criterion')
parser.add_argument('--scoring_metrics', type=str, default='accuracy', help='Classifiaction metrics used for early stopping')

# Session parameters
parser.add_argument('--restore', action='store_true', help='Restore previously trained model')
parser.add_argument('--checkpoint_file', type=str, default='../checkpoints/lstm', help='Checkpoint file path')
parser.add_argument('--allow_soft_placement', type=bool, default=True, help='Allow soft device replacement')
parser.add_argument('--log_device_placement', type=bool, default=False, help='Log placement of ops on devices')

args = parser.parse_args()

	

			
	
if __name__ == '__main__':
	t0 = time.time()
	if not args.tune:
		TRAIN(args, args.model)
	else:

		if args.tuning_method == 'skopt':
			skoptTUNE(args, args.model, args.num_calls)
		elif args.tuning_method == 'hyperopt':
			hyperoptTUNE(args, args.model, args.num_calls)
		elif args.tuning_method == 'rand':
			TUNE(args, args.model, 'rand', args.num_calls)
		else:
			TUNE(args, args.model, 'grid')
	# TEST(args.model)
	print()
	print("Total time taken: %f mins"%((time.time()-t0)/60))

