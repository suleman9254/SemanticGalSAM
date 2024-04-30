from torch.optim.lr_scheduler import LRScheduler

class Warmup(LRScheduler):
    def __init__(self, optimizer, T_max, warmup, last_epoch=-1, verbose="deprecated"):
        self.T_max = T_max
        self.WP = warmup
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.WP:
            return [self._step_count * base_lr / self.WP for base_lr in self.base_lrs]
        else:
            return [base_lr * (1 - (self._step_count - self.WP) / self.T_max) for base_lr in self.base_lrs]