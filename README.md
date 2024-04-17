# aiayn

Custom implementation of the paper "Attention Is All You Need", linked [here](https://arxiv.org/abs/1706.03762)

- ```pip install -r requirements.txt```
- Download the WMT 2014 English-German training (4.5m) and test sets:
  - ```wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en```
  - ```wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de```
- After installing ```spacy```, you'll need to download the English and German tokeniser models: 
  - ```python -m spacy download en && python -m spacy download de```
