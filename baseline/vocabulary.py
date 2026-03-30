import numpy as np

class Vocabulary:
    def __init__(self):
        # Special tokens
        self.word2idx = {"<PAD>": 0,"<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        self.vocab_size = 4

    def build_vocab(self, sentences):
        for sentence in sentences:
            for word in sentence:
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, sentence):
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sentence]
    
    def decode(self, ids):
        return [self.idx2word.get(idx, "<UNK>") for idx in ids]

            