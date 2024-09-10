import torch
import torch.nn as nn
from smol_gpt import SmolGPTModel, generate_text_simple
import tiktoken
from dataset import create_dataloader

SMOLGPT_CONFIG_124M_MCL = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 256, # Context size, reduced to be computationally lighter
    "embed_dim": 768, # Embedding size
    "n_layers": 12, # Number of layers
    "n_heads": 12, # Number of attention heads
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False, # Query-Key-Value bias
}

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

torch.manual_seed(123)
GPT = SmolGPTModel(SMOLGPT_CONFIG_124M_MCL)
GPT.eval()

tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Every effort moves you"

token_ids = generate_text_simple(model=GPT, 
                                 idx=text_to_token_ids(start_context, tokenizer), 
                                 max_new_tokens=10, 
                                 context_size=SMOLGPT_CONFIG_124M_MCL["context_length"])
print(f'Output text: {token_ids_to_text(token_ids, tokenizer)}')

# Calculate the loss function (Cross Entropy)
''' 
1. Generate logits from the model
2. Converting logits to probabilities using softmax
3. Extracting the probabilities of the target tokens
4. Calculating the log of these probabilities
5. Averaging the log probabilities
6. Negating the average log probabilities to get the loss value
'''

# Our (input, target) pairs of example
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]

# Generate logits
with torch.no_grad():
    logits = GPT(inputs)

# Converting logits to probabilities using softmax
probas = torch.softmax(logits, dim=-1)
print(probas)
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# Extracting the probabilities of the target tokens
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

# Calculating the log of these probabilities
log_probas = torch.log(torch.cat([target_probas_1, target_probas_2]))
print(log_probas)

# Averaging the log probabilities
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

# Negating the average log probabilities to get the loss value
loss = -avg_log_probas
print(f'My loss: {loss}')
print(f'PyTorch loss: {nn.CrossEntropyLoss()(logits.flatten(0, 1), targets.flatten())}')

# Perplexity is a measure of how well a probability distribution or probability model predicts a sample.
# perplexity = exp(cross_entropy)
print(f'Perplexity: {torch.exp(loss)}') # 48725, it means that the model is unsure about which among 48725 words of the vocabulary is the next token

# Now let's calculate the loss over our entire dataset
with open('the_verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

tokenized_text = tokenizer.encode(raw_text)
total_tokens = len(tokenized_text)
print(f'Total characters: {len(raw_text)}')
print(f'Total tokens: {total_tokens}')

# train/val split
train_ratio = 0.9
split_idx = int(len(raw_text) * train_ratio)
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]

torch.manual_seed(123)
train_loader = create_dataloader(train_data, tokenizer=tokenizer,
                                 max_length=SMOLGPT_CONFIG_124M_MCL["context_length"], 
                                 stride=SMOLGPT_CONFIG_124M_MCL["context_length"],
                                 batch_size=2, 
                                 drop_last=True, 
                                 shuffle=True)
val_loader = create_dataloader(val_data, tokenizer=tokenizer,
                                 max_length=SMOLGPT_CONFIG_124M_MCL["context_length"], 
                                 stride=SMOLGPT_CONFIG_124M_MCL["context_length"],
                                 batch_size=2, 
                                 drop_last=True, 
                                 shuffle=True)

print("Train loader check:")
for x, y in train_loader:
    print(x.shape, y.shape) # [2, 256] --> 2 samples and 256 tokens each

print("Val loader check:")
for x, y in val_loader:
    print(x.shape, y.shape) # [2, 256] --> 2 samples and 256 tokens each
    
# utility function to calculate che CE loss of a given batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = nn.CrossEntropyLoss()(logits.flatten(0, 1), target_batch.flatten())
    return loss

# function to compute training and validation loss
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# Let's try it on training and validation data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPT.to(device)
train_loss = calc_loss_loader(train_loader, GPT, device, num_batches=10)
val_loss = calc_loss_loader(val_loader, GPT, device, num_batches=10)
print(f'Train Loss: {train_loss}')
print(f'Val Loss: {val_loss}')