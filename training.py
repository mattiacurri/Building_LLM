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
with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, GPT, device)
    val_loss = calc_loss_loader(val_loader, GPT, device)
print(f'Train Loss: {train_loss}')
print(f'Val Loss: {val_loss}')

# Let's create a training loop
def evaluate_model(model, train_loader, val_loader, device, num_batches):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.position_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def training_loop_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context):
    train_losses, val_losses, track_token_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            
            # Optional evaluation of the model
            if global_step % eval_freq == 0:
                # Numeric estimate of the training process
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_freq)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)
                print(f'Epoch: {epoch}, Global Step: {global_step}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')
                
        # Provides a sense of how the model is doing
        generate_and_print_sample(model, train_loader.dataset.tokenizer, device, start_context)
    return train_losses, val_losses, track_token_seen

import time
start_time = time.time()

torch.manual_seed(123)
GPT = SmolGPTModel(SMOLGPT_CONFIG_124M_MCL)
GPT.to(device)
optimizer = torch.optim.AdamW(GPT.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = training_loop_simple(
    GPT, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you",
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# Let's plot the training and validation loss
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# Now let's use some decoding strategies to generate text (temperature scaling and top-k sampling)
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        # top-k sampling: maintain only the top k highest probabilities
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k, dim=-1)
            min_val = top_logits[:, -1]
            # mask with -inf to exclude the values below the top-k
            logits = torch.where(logits < min_val, torch.tensor(-float('inf')).to(logits.device), logits)
        # temperature scaling: increase the temperature to make the distribution more uniform or decrease it to make it more peaky (more confident)
        if temperature > 0.0:
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1)
        
        if next_token == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break
        idx = torch.cat((idx, next_token), dim=1)
    return idx

# test the function
torch.manual_seed(123)
GPT.eval()
GPT.to("cpu")

token_ids = generate(model=GPT, 
                     idx=text_to_token_ids("Every effort moves you", tokenizer), 
                     max_new_tokens=15, 
                     context_size=SMOLGPT_CONFIG_124M_MCL["context_length"],
                     temperature=1.4, top_k=25)
print(f'Output text: {token_ids_to_text(token_ids, tokenizer)}')