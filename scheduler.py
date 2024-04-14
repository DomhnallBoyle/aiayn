import torch

import config


class CustomScheduler(torch.optim.lr_scheduler.LRScheduler):

    def __init__(self, optimiser):
        super().__init__(optimiser)

        self.num_steps = 0

    def get_lr(self):
        return (config.d_model ** -0.5) * min(self.num_steps ** -0.5, self.num_steps * (config.warmup_steps ** -1.5))

    def step(self):
        self.num_steps += 1

        # NOTE: this calls get_lr() internally, before updating the optimiser LR
        super().step()

