import random

import torch


class CustomSampler(torch.utils.data.Sampler):
    # batches together similar sized samples - reduces padding

    def __init__(self, source_dataset, batch_size):
        self.source_dataset = source_dataset
        self.batch_size = batch_size
        self.num_batches = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # yields a batch of indices every iteration/step
        batches = self._generate_batches()  # generate random batches initially every epoch
        random.shuffle(batches)  # aren't ordered by length

        for batch in batches:
            yield batch

    def _generate_batches(self):
        length_map = self._generate_length_map()

        for indices in length_map.values():
            random.shuffle(indices)  # indices put into different batches

        batches = [indices[i: i + self.batch_size]
                   for indices in length_map.values()
                   for i in range(0, len(indices), self.batch_size)]        

        batches = [b for b in batches if len(b) == self.batch_size]  # forcing batch size
        self.num_batches = len(batches)

        return batches

    def _generate_length_map(self):
        length_map = {}  # length, indices lookup

        for i, length in self.source_dataset.lengths.items():
            indices = length_map.get(length, [])
            indices.append(i)
            length_map[length] = indices

        return length_map
