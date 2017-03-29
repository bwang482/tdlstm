import random
import itertools as it
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from utils import load_data
from train_lstm import LSTM
from train_tdlstm import TDLSTM
from train_tclstm import TCLSTM
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.space import Categorical



def random_search(param_grid, sampsize=None):

    expanded_param_grid = expand_grid(param_grid)
    if sampsize == None:
        sampsize = int(len(expanded_param_grid) / 2.0)
    samp = random.sample(expanded_param_grid, sampsize)
    return samp


def expand_grid(param_grid):       
    varNames = sorted(param_grid)
    return [dict(zip(varNames, prod))
            for prod in it.product(*(param_grid[varName]
                                     for varName in varNames))]



def skopt_search(args, data, model, param_grid, skopt_method, n_calls):

    param_keys, param_vecs = zip(*param_grid.items())
    param_keys = list(param_keys)
    param_vecs = list(param_vecs)

    def skopt_scorer(param_vec):
        params = dict(zip(param_keys, param_vec))
        args.num_hidden = params['num_hidden']
        args.dropout_output = params['dropout_output']
        args.dropout_input = params['dropout_input']
        args.clip_norm = params['clip_norm']
        args.batch_size = params['batch_size']
        print(args)
        print()
        test_score, eval_score = run_network(args, data, model, tuning=args.tune)
        tf.reset_default_graph()
        eval_score = -eval_score[0]
        return eval_score

    outcome = skopt_method(skopt_scorer, list(param_vecs), n_calls=n_calls)
    results = []
    for err, param_vec in zip(outcome.func_vals, outcome.x_iters):
        params = dict(zip(param_keys, param_vec))
        results.append({'loss': err, 'params': params})
    return results


def skoptTUNE(args, model, n_calls):
    """
    Hyper-parameter optimization using scikit-opt.
    It has 3 algorithms: forest_minimize (decision-tree regression search);
    gbrt_minimize (gradient-boosted-tree search);
    and hp_minimize (Gaussian process regression search).
    """
    hyperparameters = {
        'batch_size': (40, 120),
        'num_hidden': (100, 500),
        'dropout_output': (0.3, 1.0),
        'dropout_input': (0.3, 1.0),
        'clip_norm': (0.5, 1.0),
    }

    data = load_data(args, args.data, saved=args.load_data)
    all_res = skopt_search(args, data, model, hyperparameters, gp_minimize, n_calls=n_calls)
    print(all_res)


def hyperopt_search(args, data, model, param_grid, max_evals):

    def objective(param_grid):
        args.num_hidden = param_grid['num_hidden']
        args.dropout_output = param_grid['dropout_output']
        args.dropout_input = param_grid['dropout_input']
        args.clip_norm = param_grid['clip_norm']
        args.batch_size = param_grid['batch_size']
        # args.learning_rate = param_grid['learning_rate']
        print(args)
        print()
        test_score, eval_score = run_network(args, data, model, tuning=args.tune)
        tf.reset_default_graph()
        eval_score = -eval_score[0]
        return {'loss': eval_score, 'params': args, 'status': STATUS_OK}

    trials = Trials()
    results = fmin(
        objective, param_grid, algo=tpe.suggest,
        trials=trials, max_evals=max_evals)
    
    return results, trials.results


def hyperoptTUNE(args, model, n_calls):
    """
    Search the hyper-parameter space according to the tree of Parzen estimators;
    a Bayesian approach.
    """
    hyperparameters = {
        'batch_size': hp.choice('batch_size', range(40, 130, 20)),
        'num_hidden': hp.quniform('num_hidden', 100, 500, 1),
        # 'learning_rate': hp.choice('learning_rate', [0.0005]),
        'dropout_output': hp.quniform('dropout_output', 0.3, 1.0, 0.1),
        'dropout_input': hp.quniform('dropout_input', 0.3, 1.0, 0.1),
        'clip_norm': hp.quniform('clip_norm', 0.5, 1.0, 0.1),
    }

    data = load_data(args, args.data, saved=args.load_data)
    best_params, all_res = hyperopt_search(args, data, model, hyperparameters, max_evals=n_calls)
    print(best_params)



def TUNE(args, model, mode, n_calls=5):
    hyperparameters_all = {
        'batch_size': range(40, 130, 20),
        'seq_len': [42],
        'num_hidden': np.random.randint(100, 501, 10),
        'learning_rate': [0.0005],
        'dropout_output': np.arange(0.3, 1.1, 0.1),
        'dropout_input': np.arange(0.3, 1.1, 0.1),
        'clip_norm': np.arange(0.5, 1.01, 0.1),
    }
    
    maxx = 0
    data = load_data(args, args.data, saved=args.load_data)
    if mode == 'rand':
        samp = random_search(hyperparameters_all, n_calls) #random search
    else:
        samp = expand_grid(hyperparameters_all) #grid-search
    for hyperparameters in samp:
        print("Evaluating hyperparameters:", hyperparameters)
        for attr, value in hyperparameters.items():
            setattr(args, attr, value)
        test_score, eval_score = run_network(args, data, model, tuning=args.tune)
        if eval_score[0] > maxx:
            maxx = eval_score[0]
            best_score = test_score
            hyperparameters_best = hyperparameters
        tf.reset_default_graph()
    print()
    print("Optimisation finished..")
    print("Optimised hyperparameters:")
    with open(os.path.dirname(args.checkpoint_file)+'/checkpoint', 'w') as fp:
        fp.write('%s:"%s"\n' % ('model',args.model))
        for attr, value in sorted(hyperparameters_best.items()):
            print("{}={}".format(attr.upper(), value))
            fp.write('%s:"%s"\n' % (attr, value))
    print()
    print("Final Test Data Accuracy = {:.5f}; 3-class F1 = {:.5f}; 2-class F1 = {:.5f}"
                          .format(best_score[0], best_score[1], best_score[2]))


def TRAIN(args, model):
    print("\nParameters:")
    for attr, value in sorted(vars(args).items()):
        print("{}={}".format(attr.upper(), value))
    print()
    print("Graph initialized..")
    t1 = time.time()
    print("time taken:", t1-t0)
    print()
    data = load_data(args, args.data, saved=args.load_data)
    run_network(args, data, model, tuning=args.tune)


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




