import torch
from torchtext.data import get_tokenizer
from tqdm import tqdm

BOS_WORD = '<s>'
EOS_WORD = '</s>'
PAD_WORD = '<pad>'


class LanguageDataset(torch.utils.data.Dataset):
    
    def __init__(self, path, spacy_model):
        self.path = path
        self.sentences = []
        self.lengths = {}
        self.vocab = set()
        self.tokeniser = get_tokenizer('spacy', language=spacy_model)  # TODO: remove this w/ spacy?
    
        # TODO: save vocabs to disk

        print(f'Loading dataset: {self.path}...')

        # load vocab from unique words, cache untokenised strings and sentence lengths
        with open(self.path, 'r') as f:
            for i, line in enumerate(tqdm(f.read().splitlines())):
                tokens = self.tokenise(line)

                for token in tokens:
                    self.vocab.add(token)

                self.sentences.append(line)
                self.lengths[i] = len(tokens)

        for token in [BOS_WORD, EOS_WORD, PAD_WORD]:
            self.vocab.add(token)

        # create vocab and decoder lookups
        self.vocab = {token: i for i, token in enumerate(self.vocab)}
        self.decoder = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        sentence_encoded = [self.vocab[token] for token in self.tokenise(sentence)]
        sentence_encoded = [self.vocab[BOS_WORD]] + sentence_encoded + [self.vocab[EOS_WORD]]
        
        gt = ' '.join(sentence)

        return torch.Tensor(sentence_encoded).int(), gt

    def tokenise(self, line):
        return line.strip().lower().split()

    def vocab_size(self):
        return len(self.vocab)

    def decode(self, sentence_logits):
        # sample = [T, C] (softmax)
        sentence_logits_argmax = torch.argmax(sentence_logits, dim=1).tolist()  # [T]
        sentence_decoded = [self.decoder[i] for i in sentence_logits_argmax]

        return ' '.join(sentence_decoded)


class WMTDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.english_dataset = LanguageDataset(path='train.en', spacy_model='en_core_web_sm')
        self.german_dataset = LanguageDataset(path='train.de', spacy_model='de')

        assert len(self.english_dataset) == len(self.german_dataset)

    def __len__(self):
        return len(self.english_dataset)

    def __getitem__(self, index):
        return self.english_dataset[index], self.german_dataset[index]

    def decode(self, sample):
        return self.german_dataset.decode(sample)


if __name__ == '__main__':
    dataset = WMTDataset()
    item = dataset[0]
    
    print(item)
