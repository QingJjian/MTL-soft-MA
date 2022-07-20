import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0,  trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        self.delta = delta
        #self.path = path
        self.trace_func = trace_func
    def __call__(self, val_acc, model1,model2):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint1(val_acc, model1)
            self.save_checkpoint2(val_acc, model2)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint1(val_acc, model1)
            self.save_checkpoint2(val_acc, model2)
            self.counter = 0

    def save_checkpoint1(self, val_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Saving model1 ...')
        torch.save(model.state_dict(), 'checkpoint1.pt')
        self.val_acc_max = val_acc

    def save_checkpoint2(self, val_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Saving model2 ...')
        torch.save(model.state_dict(), 'checkpoint2.pt')
        self.val_acc_max = val_acc