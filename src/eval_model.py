"""
Helpers for evaluating models.
"""
import numpy as np
from .learners import Learner

def bulk_evaluate(learner,
            dataset,
            num_classes=5,
            num_shots=5,
            eval_inner_iters=10,
            num_samples=10000):
    
    # For evaluating the learner on a set of tasks.
    total_correct = []
    for _ in range(num_samples):
        total_correct.append(learner.evaluate(dataset, 
                        num_classes=num_classes, num_shots=num_shots,
                        inner_iters=eval_inner_iters))
    
    total_accuracies = np.array(total_correct) / num_classes
    test_accuracy = total_accuracies.sum() / num_samples
    test_cnf = np.std(total_accuracies) / np.sqrt(num_samples)

    # For confidence interval 0.95%
    # z_score = 1.96 
    # test_cnf = z_score * test_cnf 

    return test_accuracy, test_cnf
