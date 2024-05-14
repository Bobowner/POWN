import numpy as np
import torch
#Early Topping on both loss and acc
class EarlyStopping():
    def __init__(self, patience, path):
        self.patience = patience
        self.path = path
        self.counter = 0

        self.val_loss_min = np.Inf
        self.val_acc_max = -1.0

    def __call__(self, val_acc, model):
    #todo loss/acc option
        acc_worse = val_acc <= self.val_acc_max
        if acc_worse:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            if not acc_worse:
                self.val_acc_max = val_acc
            self.counter = 0
            self.save_checkpoint(model)
        return False

    def save_checkpoint(self,  model):
        torch.save(model.state_dict(), str(self.path))