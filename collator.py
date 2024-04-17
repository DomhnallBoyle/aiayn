import torch

import dataset


class CustomCollator:
    
    def __init__(self, source_vocab, target_vocab):
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __call__(self, batch):
        source, source_gt = zip(*[x[0] for x in batch])
        target, target_gt = zip(*[x[1] for x in batch])

        source = list(source)
        source_gt = list(source_gt)
        target = list(target)
        target_gt = list(target_gt)

        # pad to max len in either the source or target sentences
        max_len = max(max([len(x) for x in source]), max([len(x) for x in target]))

        source_pad_value, target_pad_value = self.source_vocab[dataset.PAD_WORD], self.target_vocab[dataset.PAD_WORD]

        # pad first seq to desired length
        source[0] = torch.nn.ConstantPad1d((0, max_len - source[0].shape[0]), source_pad_value)(source[0])
        target[0] = torch.nn.ConstantPad1d((0, max_len - target[0].shape[0]), target_pad_value)(target[0])

        # pad all seqs to desired length
        source = torch.nn.utils.rnn.pad_sequence(source, batch_first=True, padding_value=self.source_vocab[dataset.PAD_WORD])
        target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=self.target_vocab[dataset.PAD_WORD])
        
        return source, target, source_gt, target_gt
