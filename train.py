import argparse

import torch

import config
from collator import CustomCollator
from dataset import WMTDataset
from model import Transformer
from sampler import CustomSampler
from scheduler import CustomScheduler
from utils import get_num_params


def main(args):

    dataset = WMTDataset()

    model = Transformer(
        source_vocab_size=dataset.english_dataset.vocab_size(),
        target_vocab_size=dataset.german_dataset.vocab_size()
    ).to(config.device)
    
    optimiser = torch.optim.Adam(
        model.parameters(), 
        lr=config.lr_initial, 
        betas=config.lr_betas, 
        eps=config.lr_eps
    )
    lr_scheduler = CustomScheduler(optimiser)

    # apply label smoothing - targets become mixture of original gt and a uniform distribution
    # restrains the largest logit from becoming much bigger than the rest
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    sampler = CustomSampler(
        source_dataset=dataset.english_dataset, 
        batch_size=args.batch_size
    )
    collator = CustomCollator(
        source_vocab=dataset.english_dataset.vocab,
        target_vocab=dataset.german_dataset.vocab
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_sampler=sampler,
        collate_fn=collator,
        pin_memory=True
    )

    model.train()
    model.zero_grad()

    epoch = 0
    num_steps = 0
    running_loss = 0
    finished_training = False

    # TODO: gradient accumulation?

    print(model)
    print('Num model params:', get_num_params(model))
    print('Num dataset samples:', len(dataset))
    print(get_num_params(model.encoder.embedding_layer))

    while not finished_training:
        for i, train_data in enumerate(train_loader):
            source, target, source_gt, target_gt = train_data  # batch

            source = source.to(config.device)
            target = target.to(config.device)

            output = model(source, target)  # [B, T, C]

            # input = model prediction [B, C, T]
            # target = label tensor [B, T], long
            loss = criterion(output.permute(0, 2, 1), target.long())
            loss.backward()  # accumulates the gradients from every forward pass

            optimiser.step()  # update weights
            optimiser.zero_grad()  # only zero the gradients after every update

            lr_scheduler.step()  # adjusting lr

            num_steps += 1
            if num_steps == args.training_steps:
                finished_training = True
                break

            running_loss += loss.item()

            if num_steps % args.log_every == 0:
                print(f'Epoch: {epoch}, Steps: {num_steps}, Av. Loss: {running_loss / args.log_every}, LR: {lr_scheduler.lr}')

                # output training sample
                print(f'Input: {source_gt[0]}\nTarget: {target_gt[0]}\nOutput: {dataset.decode(output[0])}\n')

                # reset running variables
                running_loss = 0

        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int)
    parser.add_argument('training_steps', type=int)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)

    main(parser.parse_args())
