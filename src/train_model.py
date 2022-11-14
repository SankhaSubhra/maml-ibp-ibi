import os
import numpy as np

import torch
from .learners import Learner

def train(learner,
        train_set,
        val_set,
        model_output_file=None,
        model_save_path=None,
        order=None,
        num_classes=None,
        num_shots=None,
        meta_shots=None,
        inner_iters=None,
        meta_batch_size=None,
        meta_iters=None,
        eval_inner_iters=None,
        eval_interval=None,
        eval_interval_sample=None,
        ibp_epsilon=None,
        mixup=False,
        ibp_layers=None):
    
    # Train a model on a dataset.
    train_accuracy, val_accuracy = [], []
    upper_loss_store, lower_loss_store, task_loss_store, total_loss_store = [], [], [], []

    # Loop over the training steps.
    for i in range(meta_iters+1):
        
        # Find current value of interval coefficient.
        cur_ibp_epsilon = eps_scheduler(i, meta_iters, ibp_epsilon)

        # Train the learner for a step.
        upper_loss, lower_loss, task_loss, total_loss = learner.train_step(train_set, order=order,
                                                            num_classes=num_classes, num_shots=num_shots,
                                                            meta_shots=meta_shots,
                                                            inner_iters=inner_iters,
                                                            meta_batch_size=meta_batch_size,
                                                            ibp_epsilon=cur_ibp_epsilon,
                                                            mixup=mixup,
                                                            ibp_layers=ibp_layers)
        # Record losses
        upper_loss_store.append(upper_loss)
        lower_loss_store.append(lower_loss)
        task_loss_store.append(task_loss)
        total_loss_store.append(total_loss)
        
        if i % eval_interval == 0:
            
            # Perform intermediate evaluation.
            total_correct = 0
            for _ in range(eval_interval_sample):
                total_correct = total_correct + learner.evaluate(train_set,
                                                    num_classes=num_classes, num_shots=num_shots,
                                                    inner_iters=eval_inner_iters)
            
            train_accuracy.append(total_correct / (eval_interval_sample * num_classes))

            save_path = model_save_path + '/intermediate_' + str(i) + '_model.pt'

            torch.save({'model_state': learner.net.state_dict(),
                        'meta_optim_state': learner.meta_optim.state_dict()},
                       save_path)

            total_correct = 0
            for _ in range(eval_interval_sample):
                total_correct = total_correct + learner.evaluate(val_set,
                                                    num_classes=num_classes, num_shots=num_shots,
                                                    inner_iters=eval_inner_iters)
            val_accuracy.append(total_correct / (eval_interval_sample * num_classes))

            with open(model_output_file, 'a+') as fp:
                print('batch %d: train=%f val=%f' % (i, 
                    train_accuracy[-1], val_accuracy[-1]), file=fp)

    # Intermediate record keeping.
    res_save_path = model_save_path + '/' + 'intermediate_accuracies.npz'
    loss_save_path = model_save_path + '/' + 'intermediates_losses.npz'

    np.savez(res_save_path, train_accuracy=np.array(train_accuracy),
        val_accuracy=np.array(val_accuracy))
    
    np.savez(loss_save_path, upper_loss=np.array(upper_loss_store),
        lower_loss=np.array(lower_loss_store), task_loss=np.array(task_loss_store),
        total_loss=np.array(total_loss_store))

def eps_scheduler(i, meta_iters, ibp_epsilon):
    
    # Schedule the value of interval coefficient.

    if i < meta_iters*0.9:
        return (i / (meta_iters*0.9)) * ibp_epsilon

    return ibp_epsilon

        
