# the dataset is a single file to be able to run all the code in the repo on consumer devices
with open('the_verdict.txt', 'r') as file:
    raw_text = file.read()
print(f'Number of characters: {len(raw_text)}')

# TOKENIZATION STEPS: raw_text -> tokenization -> vocab -> token ids -> embedding

# tokenizing with a simple regex
import re
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip() != '']
print(f'Number of tokens: {len(preprocessed)}')

# creating the vocabulary
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(['<|unk|>', '<|endoftext|>'])
vocab_size = len(all_tokens)
print(f'Vocab size: {vocab_size}')
vocab = {token: i for i, token in enumerate(all_tokens)}

# now we can define a simple tokenizer class
from SimpleTokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer(vocab)
text1 = "The fox jumps over the lazy dog"
text2 = "You are my sunshine, my only sunshine"
text3 = " <|endoftext|> ".join([text1, text2])
print(text3)
ids = tokenizer.encode(text3)
print(ids)
print(tokenizer.decode(ids))

# now we will use byte pair encoding (BPE), which is a kind of tokenization that is used by openAI's models
import tiktoken # written by tiktoken, it contains the BPE algorithm

gpt_tokenizer = tiktoken.get_encoding("o200k_base") # the encoding used for gpt-4o according to a sketchy Go documentation of tiktoken I found online

# print # of vocab
print(f'Vocab size of gpt-4o: {gpt_tokenizer.n_vocab}')

# usage similar to the tokenizer implemented above
ids = gpt_tokenizer.encode(text3, allowed_special={'<|endoftext|>'})

print(ids)
print(gpt_tokenizer.decode(ids))