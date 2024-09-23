import torch
import torch.nn as nn
import tiktoken
import time

import os
import sys

# add in sys.path the path of the folder containing the module
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smolgpt.SmolGPT import SmolGPTModel, generate_text_simple, generate, text_to_token_ids, token_ids_to_text
from dataset.SimpleDataset import create_dataloader
from smolgpt.gpt2 import SMOLGPT_CONFIG_124M_MCL

from training_loop import calc_loss_loader, training_loop_simple, plot_losses

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
# perplexity = exp(loss)
print(f'Perplexity: {torch.exp(loss)}') # 48725, it means that the model is unsure about which among 48725 words of the vocabulary is the next token

# Now let's calculate the loss over our entire dataset
with open('dataset/the_verdict.txt', 'r', encoding='utf-8') as f:
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
train_loader = create_dataloader(
    train_data,
    tokenizer,
    batch_size=2,
    max_length=SMOLGPT_CONFIG_124M_MCL["context_length"],
    stride=SMOLGPT_CONFIG_124M_MCL["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader(
    val_data,
    tokenizer,
    batch_size=2,
    max_length=SMOLGPT_CONFIG_124M_MCL["context_length"],
    stride=SMOLGPT_CONFIG_124M_MCL["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

print("Train loader check:")
for x, y in train_loader:
    print(x.shape, y.shape) # [2, 256] --> 2 samples and 256 tokens each

print("Val loader check:")
for x, y in val_loader:
    print(x.shape, y.shape) # [2, 256] --> 2 samples and 256 tokens each

train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)


# Let's try it on training and validation data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPT.to(device)
with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, GPT, device)
    val_loss = calc_loss_loader(val_loader, GPT, device)
print(f'Train Loss: {train_loss}')
print(f'Val Loss: {val_loss}')

start_time = time.time()

torch.manual_seed(123)
GPT = SmolGPTModel(SMOLGPT_CONFIG_124M_MCL)
GPT.to(device)
optimizer = torch.optim.AdamW(GPT.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = training_loop_simple(
    GPT, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

torch.save({"model_state_dict": GPT.state_dict(), 
            "optimizer_state_dict": optimizer.state_dict()}, 
           "to_ignore/SmolGPT.pth")

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, xlabel="Tokens seen", file_name="to_ignore/loss-plot.pdf")

torch.manual_seed(123)
GPT.eval()
GPT.to("cpu")

token_ids = generate(model=GPT, 
                     idx=text_to_token_ids("Every effort moves you", tokenizer), 
                     max_new_tokens=15, 
                     context_size=SMOLGPT_CONFIG_124M_MCL["context_length"],
                     temperature=1.4, top_k=25)
print(f'Output text: {token_ids_to_text(token_ids, tokenizer)}')

# Let's test the better training loop
# start_time = time.time()

# torch.manual_seed(123)
# model = SmolGPTModel(SMOLGPT_CONFIG_124M_MCL)
# model.to(device)

# peak_lr = 5e-4
# optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)

# n_epochs = 15
# train_losses, val_losses, tokens_seen, lrs = better_training_loop(
#     model, train_loader, val_loader, optimizer, device, num_epochs=n_epochs,
#     eval_freq=5, eval_iter=1, start_context="Every effort moves you",
#     warmup_steps=20, initial_lr=1e-5, min_lr=1e-5
# )

# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60
# print(f"Training completed in {execution_time_minutes:.2f} minutes.")