import random
import os 
import sys
import numpy as np
import torch

import argparse
from img_data_process import read_dataset
from datetime import datetime
from copy import deepcopy

from src.models import NetworkModel
from src.eval_model import bulk_evaluate
from src.train_model import train
from src.learners import Learner

def argument_parser():
    # Get an argument parser for a training script.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', help='name of dataset', default=None)
    parser.add_argument('--algorithm', help='name of algorithm', default=None)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--order', help='order for MAML and variants.', default=None, type=int)
    parser.add_argument('--classes', help='number of classes per inner task', default=None, type=int)
    parser.add_argument('--shots', help='number of examples per class', default=None, type=int)
    parser.add_argument('--meta-shots', help='shots for meta update', default=None, type=int)
    parser.add_argument('--inner-iters', help='inner iterations', default=None, type=int)
    parser.add_argument('--learning-rate',help='inner loop learning rate', default=None, type=float)
    parser.add_argument('--meta-step', help='outer loop learning rate', default=None, type=float)
    parser.add_argument('--meta-batch', help='meta-training batch size', default=None, type=int)
    parser.add_argument('--meta-iters', help='meta-training iterations', default=None, type=int)
    parser.add_argument('--eval-iters', help='evaluation inner iterations', default=None, type=int)
    parser.add_argument('--eval-samples', help='evaluation samples', default=None, type=int)
    parser.add_argument('--eval-interval',help='evaluation interval during training', default=None, type=int)
    parser.add_argument('--eval-interval-sample',help='evaluation samples during training', default=None, type=int)
    parser.add_argument('--ibp-eps', help='IBP neighborhood size', default=0, type=float)
    parser.add_argument('--softmax-temp', help='softmax temperature', default=None, type=float)
    parser.add_argument('--only-evaluation', help='for only evaluation', action='store_true', default=False)
    parser.add_argument('--checkpoint', help='load saved checkpoint from path', default=None)
    parser.add_argument('--test-iters', help='number of evaluations', default=None, type=int)
    parser.add_argument('--beta-a', help='beta distrebution parameter a', default=None, type=float)
    parser.add_argument('--beta-b', help='beta distribution parameter b', default=None, type=float)
    parser.add_argument('--mixup', help='set to use mixup task', action='store_true', default=False)
    parser.add_argument('--ibp-layers', help='number layer to perform IBP/IBI', default=None, type=int)
    return parser

def model_kwargs(parsed_args):
    
    # Parameters used for initializing the learner.

    return {
        'update_lr': parsed_args.learning_rate,
        'meta_step_size': parsed_args.meta_step,
        'beta_a': parsed_args.beta_a,
        'beta_b': parsed_args.beta_b,
        'softmax_temp': parsed_args.softmax_temp
    }

def train_kwargs(parsed_args):
    
    # Parameters used for training.

    return {
        'order': parsed_args.order,
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'meta_shots': parsed_args.meta_shots,
        'inner_iters': parsed_args.inner_iters,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
        'eval_interval_sample': parsed_args.eval_interval_sample,
        'ibp_epsilon': parsed_args.ibp_eps,
        'mixup': parsed_args.mixup,
        'ibp_layers': parsed_args.ibp_layers
    }

def evaluate_kwargs(parsed_args):
    
    # Parameters used for evaluation over multiple tasks.

    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'eval_inner_iters': parsed_args.eval_iters,
        'num_samples': parsed_args.eval_samples
    }

def main():
    
    args = argument_parser().parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Edit here according to need.
    DATA_DIR = '' + args.dataset

    # Create directory for storing results and initiate logging.
    if os.path.exists(os.path.join(DATA_DIR, 'val')):
        val_presence = True
        print("Validation set is present.")
    else:
        val_presence = False
        print("Validation set is not found. Exiting.")
        sys.exit()

    time_string = datetime.now().strftime("%m%d%Y_%H:%M:%S")
    output_folder = args.dataset + '_' + args.algorithm + '_output_folder_' + time_string
    output_file = output_folder + '/' + 'log_' + time_string + '.txt'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(output_file, 'a+') as fp:
        print('\n'.join(f'{k}={v}' for k, v in vars(args).items()), file=fp)

    device = torch.device('cuda')

    # Instantiate the dataset.
    train_set, val_set, test_set = read_dataset(DATA_DIR, val_presence)

    # Instantiate the learner
    model=NetworkModel(args.classes)

    learner = Learner(model, device, **model_kwargs(args))

    # Perform training or evaluation as per need.
    if args.only_evaluation is False:
        train(learner, train_set, val_set, output_file, output_folder, **train_kwargs(args))
    else:
        assert args.checkpoint is not None, 'For evaluating without training please provide a checkpoint'
        print('Evaluating...')
        
        res_file = output_folder + '/' + 'test_performance_' + time_string + '_.txt'
        with open(res_file, 'a+') as fp:
            print('Evalulation checkpoint: ' + args.checkpoint, file=fp)

        checkpoint_model = torch.load(args.checkpoint, map_location='cuda:0')
        learner.net.load_state_dict(checkpoint_model['model_state'])
        learner.meta_optim.load_state_dict(checkpoint_model['meta_optim_state'])

        train_accuracy, val_accuracy, test_accuracy = [], [], []
        train_cnf, val_cnf, test_cnf = [], [], []

        for ii in range(args.test_iters):

            train_acc, train_div = bulk_evaluate(learner, train_set, **evaluate_kwargs(args))
            val_acc, val_div = bulk_evaluate(learner, val_set, **evaluate_kwargs(args))
            test_acc, test_div = bulk_evaluate(learner, test_set, **evaluate_kwargs(args))
            
            train_accuracy.append(train_acc)
            val_accuracy.append(val_acc)
            test_accuracy.append(test_acc)

            train_cnf.append(train_div)
            val_cnf.append(val_div)
            test_cnf.append(test_div)

            with open(res_file, 'a+') as fp:
                print('Test iteration: ' + str(ii + 1), file=fp)
                print('Train accuracy: ' + str(train_accuracy[-1]) + ' +/- ' + str(train_cnf[-1]), file=fp)
                print('Validation accuracy: ' + str(val_accuracy[-1]) + ' +/- ' + str(val_cnf[-1]), file=fp)
                print('Test accuracy: ' + str(test_accuracy[-1]) + ' +/- ' + str(test_cnf[-1]) + '\n', file=fp)

        save_path = output_folder + '/' + 'results' + '.npz'
        train_accuracy = np.array(train_accuracy)
        val_accuracy = np.array(val_accuracy)
        test_accuracy = np.array(test_accuracy)

        train_cnf = np.array(train_cnf)
        val_cnf = np.array(val_cnf)
        test_cnf = np.array(test_cnf)
        
        np.savez(save_path, train_accuracy=train_accuracy, val_accuracy=val_accuracy, 
            test_accuracy=test_accuracy, train_confidence=train_cnf, val_confidence=val_cnf, 
            test_confidence=test_cnf)

if __name__ == '__main__':
    main()
