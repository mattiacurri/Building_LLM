# from tokenization import gpt_tokenizer as tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import os

class SimpleDataset(Dataset):
    def __init__(self, raw_text, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.targets = []
        
        token_ids = tokenizer.encode(raw_text) # encode the text into tokens

        # we will create a dataset with pairs of input_ids (the context given to the llm) and targets (the next token in the sequence)
        for i in range(0, len(token_ids) - max_length, stride):
            context = token_ids[i:i+max_length]
            target = token_ids[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(context))
            self.targets.append(torch.tensor(target))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.targets[idx]


def create_dataloader(raw_text, tokenizer, max_length=256, stride=128, batch_size=4, shuffle=True, drop_last=True, num_workers=0):
    dataset = SimpleDataset(raw_text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader


