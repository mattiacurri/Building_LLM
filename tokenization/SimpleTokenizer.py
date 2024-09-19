import re

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: token for token, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip() != '']
        preprocessed = [token if token in self.str_to_int else '<|unk|>' for token in preprocessed]
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids
    
    def decode(self, ids):
        text = ' '.join([self.int_to_str[id] for id in ids])
        text = re.sub(r'\s+([,.?_!"()\'])', r'\1', text) # add spaces around punctuation
        return text