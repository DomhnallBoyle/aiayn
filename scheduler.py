import torch

import config


class CustomScheduler(torch.optim.lr_scheduler.LRScheduler):

    def __init__(self, optimiser, d_model=config.d_model, warmup_steps=config.warmup_steps):
        self.optimiser = optimiser
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.num_steps = 0
        self.lr = None

        super().__init__(optimiser)

    def get_lr(self):
        self.lr = (self.d_model ** -0.5) * min(self.num_steps ** -0.5, self.num_steps * (self.warmup_steps ** -1.5))

        return [self.lr for _ in range(len(self.optimizer.param_groups))]

    def step(self):
        self.num_steps += 1

        # NOTE: this calls get_lr() internally, before updating the optimiser LR
        super().step()
