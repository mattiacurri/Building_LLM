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

with open("the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# stride 4 to fully utilize the data set and avoid overlap and (maybe) overfitting
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

max_length = 4
dataloader = create_dataloader(
    raw_text, tokenizer, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False, num_workers=os.cpu_count()
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
    
torch.manual_seed(42)
#embedding = torch.nn.Embedding(6, 3) # row: len(#token), column: embedding dimension
#print(embedding.weight)
#print(embedding(torch.tensor([4, 3, 5, 1]))) # the embedding layer is essentially a look-up operation that retrieves rows from the embedding matrix via a token id


# now let's create our token embeddings using absolute positional embeddings
output_dim = 256
vocab_size = tokenizer.n_vocab
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(f'Shape of Token Embedding Layer: {token_embedding_layer.weight.shape}') # [vocab_size, output_dim]

token_embeddings = token_embedding_layer(inputs)
print(f'Shape of Token Embeddings: {token_embeddings.shape}') # [batch_size, max_length, output_dim]

# now let's add the positional embeddings
context_length = token_embeddings.shape[1] # max_length
position_ids = torch.arange(context_length) # placeholder vector for the position ids, contains a sequence of integers from 0 to max_length - 1
position_embedding_layer = torch.nn.Embedding(context_length, output_dim)
position_embeddings = position_embedding_layer(position_ids)
print(f'Shape of Position Embedding Layer: {position_embedding_layer.weight.shape}') # [max_length, output_dim]

# input_embeddings are the embedded input examples that can now be processed by our LLM! 🚀
input_embeddings = token_embeddings + position_embeddings # add the positional embeddings to the token embeddings
print(f'Shape of Position and Token Embeddings: {input_embeddings.shape}') # [batch_size, max_length, output_dim]
