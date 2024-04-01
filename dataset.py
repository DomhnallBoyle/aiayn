import torch
from torchtext.data import get_tokenizer

BOS_WORD = '<s>'
EOS_WORD = '</s>'
PAD_WORD = '<pad>'


class LanguageDataset(torch.utils.data.Dataset):
    
    def __init__(self, path, spacy_model):
        self.path = path
        self.sentences = []
        self.vocab = set()
        self.tokeniser = get_tokenizer('spacy', language=spacy_model)
    
        with open(self.path, 'r') as f:
            for line in f.read().splitlines():
                tokens = self.tokeniser(line.lower())
                self.sentences.append(tokens)

        for tokens in self.sentences:
            for token in tokens:
                self.vocab.add(token)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        pass

    def vocab_size(self):
        return len(self.vocab)


class WMTDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.english_dataset = LanguageDataset(path='dev/newstest2013.en', spacy_model='en_core_web_sm')
        # self.german_dataset = LanguageDataset(path='dev/newstest2013.de', language='de')

        # assert len(self.english_dataset) == len(self.german_dataset)

    def __len__(self):
        return len(self.english_dataset)

    def __getitem__(self, index):
        pass

