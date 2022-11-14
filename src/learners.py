import random

import torch
import torch.nn.functional as F
from torch import optim
import numpy as np

import torch.optim as optim
from copy import deepcopy
from collections import OrderedDict

class Learner:

    # Base Learner class for MAML and IBP/IBI variants.

    def __init__(self, model, device, update_lr, meta_step_size, beta_a, beta_b, softmax_temp):

        # Initialization.
        self.device = device
        self.net = model.to(self.device)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=meta_step_size)
        self.update_lr = update_lr
        self.beta_a, self.beta_b = beta_a, beta_b
        self.softmax_temp = softmax_temp

    def train_step(self,
                dataset,
                order,
                num_classes,
                num_shots,
                meta_shots,
                inner_iters,
                meta_batch_size,
                ibp_epsilon,
                mixup,
                ibp_layers):
        
        # Training function for MAML, MAML+IBP, and MAML+IBI learners.

        # For record keeping
        upper_loss_rec, lower_loss_rec, task_loss_rec, total_loss_rec = 0, 0, 0, 0

        # Triigers FOMAML and variants if required.
        create_graph, retain_graph = True, True
        if order == 1:
            create_graph, retain_graph = False, False

        self.meta_optim.zero_grad()

        if mixup is True:
            random_task = np.random.rand(0, meta_batch_size)

        # Iterate over tasks in a meta-batch
        for task_ind in range(meta_batch_size):
            
            fast_weight = OrderedDict(self.net.named_parameters())

            train_set, test_set = _split_train_test(
                _sample_mini_dataset(dataset, num_classes, num_shots+meta_shots), test_shots=meta_shots)

            # Support set
            inputs, labels = zip(*train_set)
            inputs = torch.stack(inputs).to(self.device)
            labels = torch.tensor(labels).to(self.device)

            # Fix ordering
            labels, sort_index = torch.sort(labels)
            inputs = inputs[sort_index]
            
            inputs_cat = torch.cat([inputs+ibp_epsilon, inputs, inputs-ibp_epsilon], 0)

            # Fast adaptation steps
            for _ in range(inner_iters):

                if mixup is True and task_ind == random_task:
                    # For MAML+IBI
                    _, logits = self.net.functional_forward(inputs_cat, fast_weight, 
                        ibp_layers, mixup, num_classes, self.beta_a, self.beta_b)
                
                    b_size = logits.shape[0]//2
                    logits_o = logits[:b_size]
                    logits_ulo = logits[b_size:]

                    fast_loss = (F.cross_entropy(logits_o, labels) + 
                        F.cross_entropy(logits_ulo, labels))/2
                else:
                    # For MAML and MAML+IBP
                    _, logits = self.net.functional_forward(inputs, fast_weight, 
                        None, False, None, None, None)

                    fast_loss = F.cross_entropy(logits, labels)

                fast_gradients = torch.autograd.grad(fast_loss, fast_weight.values(),
                    create_graph=create_graph)

                fast_weight = OrderedDict(
                    (name, param - self.update_lr * grad_param)
                    for ((name, param), grad_param) in zip(fast_weight.items(), fast_gradients))
            
            # Query set
            inputs, labels = zip(*test_set)
            inputs = torch.stack(inputs).to(self.device)
            labels = torch.tensor(labels).to(self.device)

            # Fix ordering
            labels, sort_index = torch.sort(labels)
            inputs = inputs[sort_index]

            if ibp_layers is None:
                # Vanilla MAML
                _, logits = self.net.functional_forward(inputs, fast_weight, 
                    None, False, None, None, None)
                total_loss = F.cross_entropy(logits, labels)
                task_loss_rec = task_loss_rec + total_loss.item()
                total_loss_rec = total_loss_rec + total_loss.item()

            else:
                # For MAML+IBP and MAML+IBI
                inputs_cat = torch.cat([inputs+ibp_epsilon, inputs, inputs-ibp_epsilon], 0)

                if mixup is True and task_ind == random_task:
                    # For MAML+IBI
                    ibp_estimate, logits = self.net.functional_forward(
                        inputs_cat, fast_weight, ibp_layers, mixup, num_classes, 
                        self.beta_a, self.beta_b)
                    
                    b_size = logits.shape[0]//2
                    logits_o = logits[:b_size]
                    logits_ulo = logits[b_size:]

                    task_loss = (F.cross_entropy(logits_o, labels) + 
                        F.cross_entropy(logits_ulo, labels))/2   
                else:
                    # For MAML+IBP
                    ibp_estimate, logits = self.net.functional_forward(
                        inputs_cat, fast_weight, ibp_layers, False, None, None, None)

                    task_loss = F.cross_entropy(logits, labels)

                # Find the propagated bounds
                b_size = ibp_estimate.shape[0]//3

                ibp_estimate_u = ibp_estimate[:b_size]
                ibp_estimate_o = ibp_estimate[b_size:2*b_size]
                ibp_estimate_l = ibp_estimate[2*b_size:]

                # Calculate $\mathcal{L}_{UB}$ and $\mathcal{L}_{LB}.
                upper_loss = F.mse_loss(ibp_estimate_u, ibp_estimate_o)
                lower_loss = F.mse_loss(ibp_estimate_l, ibp_estimate_o)

                # Dynamic weighting of losses
                concat_loss = torch.cat([task_loss.unsqueeze(0),
                    upper_loss.unsqueeze(0), lower_loss.unsqueeze(0)], 0)

                weights = F.softmax(concat_loss/self.softmax_temp, dim=0)
                total_loss = torch.sum(concat_loss * weights)
                
                # Record keeping
                upper_loss_rec = upper_loss_rec + upper_loss.item()
                lower_loss_rec = lower_loss_rec + lower_loss.item()
                task_loss_rec = task_loss_rec + task_loss.item()
                total_loss_rec = total_loss_rec + total_loss.item()
            
            total_loss.backward(retain_graph=retain_graph)

        # Averaging the loss over meta batches
        for params in self.net.parameters():
            params.grad = params.grad / meta_batch_size

        # update the meta learner parameters
        self.meta_optim.step()
        self.meta_optim.zero_grad()

        upper_loss_rec = upper_loss_rec/meta_batch_size
        lower_loss_rec = lower_loss_rec/meta_batch_size
        task_loss_rec = task_loss_rec/meta_batch_size
        total_loss_rec = total_loss_rec/meta_batch_size

        return upper_loss_rec, lower_loss_rec, task_loss_rec, total_loss_rec

    def evaluate(self,
                dataset,
                num_classes,
                num_shots,
                inner_iters):

        # Run a single evaluation of the model.

        # Preserve currently trained model.
        old_state = deepcopy(self.net.state_dict())
        fast_weight = OrderedDict(self.net.named_parameters())

        train_set, test_set = _split_train_test(_sample_mini_dataset(dataset, num_classes, num_shots+1))

        # Support set
        inputs, labels = zip(*train_set)
        inputs = (torch.stack(inputs)).to(self.device)
        labels = (torch.tensor(labels)).to(self.device)

        # Fast adaptation
        for _ in range(inner_iters):

            _, logits = self.net.functional_forward(inputs, fast_weight, 
                None, False, None, None, None)
            fast_loss = F.cross_entropy(logits, labels)
            fast_gradients = torch.autograd.grad(fast_loss, fast_weight.values())

            fast_weight = OrderedDict(
                (name, param - self.update_lr * grad_param)
                for ((name, param), grad_param) in zip(fast_weight.items(), fast_gradients))

        # Query set
        inputs, labels = zip(*test_set)
        inputs = (torch.stack(inputs)).to(self.device)
        labels = (torch.tensor(labels)).to(self.device)

        # Inference
        _, logits = self.net.functional_forward(inputs, fast_weight, 
            None, False, None, None, None)
        test_preds = (F.softmax(logits, dim=1)).argmax(dim=1)

        # Accuracy
        num_correct = torch.eq(test_preds, labels).sum()

        # Return network to original state for safety.
        self.net.load_state_dict(old_state)

        return num_correct.item()

def _sample_mini_dataset(dataset, num_classes, num_shots):
    
    # Sample a few shot task from a dataset.

    shuffled = list(dataset)
    random.shuffle(shuffled)
    for class_idx, class_obj in enumerate(shuffled[:num_classes]):
        for sample in class_obj.sample(num_shots):
            yield (sample, class_idx)

def _mini_batches(samples, batch_size, num_batches):
    
    # Generate mini-batches from some data.

    samples = list(samples)
    cur_batch = []
    batch_count = 0
    while True:
        random.shuffle(samples)
        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return

def _split_train_test(samples, test_shots=1):
    
    # Split a few-shot task into a train and a test set.

    train_set = list(samples)
    test_set = []
    labels = set(item[1] for item in train_set)
    for _ in range(test_shots):
        for label in labels:
            for i, item in enumerate(train_set):
                if item[1] == label:
                    del train_set[i]
                    test_set.append(item)
                    break
    if len(test_set) < len(labels) * test_shots:
        raise IndexError('not enough examples of each class for test set')
    return train_set, test_set
