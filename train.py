import argparse

import torch

import config
from collator import CustomCollator
from dataset import WMTDataset
from model import Transformer
from scheduler import CustomScheduler


def main(args):
    dataset = WMTDataset()
    model = Transformer(
        source_vocab_size=len(dataset.english_dataset.vocab),
        target_vocab_size=len(dataset.german_dataset.vocab)
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr_initial, betas=config.lr_betas, eps=config.lr_eps)
    lr_scheduler = CustomScheduler(optimiser)
    criterion = torch.nn.CrossEntropyLoss()

    collator = CustomCollator(
        source_vocab=dataset.english_dataset.vocab,
        target_vocab=dataset.german_dataset.vocab
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )

    model.train()
    model.zero_grad()

    num_steps = 0
    finished_training = False

    while not finished_training:
        for i, train_data in enumerate(train_loader):
            source, target = train_data
            print(source.shape, target.shape)

            output = model(source, target)
            print(output.shape)

            loss = criterion(output, target)

            loss.backward()
            optimiser.step()
            lr_scheduler.step()  # adjusting lr
            optimiser.zero_grad()

            num_steps += 1
            if num_steps == args.training_steps:
                finished_training = True
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int)
    parser.add_argument('training_steps', type=int)
    parser.add_argument('--num_workers', type=int, default=1)

    main(parser.parse_args())
