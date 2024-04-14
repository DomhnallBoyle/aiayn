import argparse

import torch

from dataset import WMTDataset
from model import Transformer
from scheduler import CustomScheduler


def main(args):
    dataset = WMTDataset()
    model = Transformer()
    optimiser = torch.optim.Adam(model.parameters(), lr=None, betas=config.lr_betas, eps=config.lr_eps)
    lr_scheduler = CustomScheduler(optimiser)
    criterion = torch.nn.CrossEntropyLoss()

    sampler = None
    collator = None
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, batch_sampler=sampler, collate_fn=collator, pin_memory=True)

    num_steps = 0
    finished_training = False

    while not finished_training:
        for i, train_data in enumerate(train_loader):
            source, target = train_data

            output = model(source, target)
            print(output.shape)
    
            num_steps += 1
            if num_steps == args.training_steps:
                finished_training = True
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int)
    parser.add_argument('training_steps', type=int)

    main(parser.parse_args())

