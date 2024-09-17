# We will perform a spam-not spam classification for sms messages using the UCI SMS Spam Collection dataset.
'''
Classification Finetuning
Stage 1: Dataset Preparation
1) Download the dataset
2) Preprocess the dataset
3) Create dataloaders

Stage 2: Model Setup
4) Initialize model
5) Load pretrained weights
6) Modify model for fine-tuning
7) Implement evaluation utilities

Stage 3: Model fine-tuning and usage
8) Fine-tune model
9) Evaluate fine-tuned model
10) Use model on new data
'''

# Libraries
import urllib.request
import zipfile
import os
from pathlib import Path

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken

from smol_gpt import SmolGPTModel
from gpt2 import gpt2_huggingface, BASE_CONFIG, load_weights, model_configs
from training import generate, token_ids_to_text, text_to_token_ids

# Stage 1: Dataset Preparation
# 1) Download the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
zip_path = "to_ignore/smsspamcollection.zip"
extracted_dir = "to_ignore/sms_spam_collection"
data_file_path = Path(extracted_dir) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_dir, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    with urllib.request.urlopen(url) as response, open(zip_path, "wb") as out_file:
        data = response.read()
        out_file.write(data)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_dir)
    original_file_path = Path(extracted_dir) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and extracted to {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_dir, data_file_path)

# 2) Preprocess the dataset
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

print(df["Label"].value_counts())

def undersample_data(df, label_column, target_count):
    print(f'Undersampling to {target_count} samples')
    spam = df[df[label_column] == "spam"]
    not_spam = df[df[label_column] == "ham"]
    not_spam = not_spam.sample(target_count, random_state=123)
    return pd.concat([spam, not_spam], axis=0)

balanced_df = undersample_data(df, "Label", df["Label"].value_counts()["spam"])

balanced_df["Label"] = balanced_df["Label"].apply(lambda x: 1 if x == "spam" else 0)

def train_valid_test_split(df, train_frac, validation_frac):
    # shuffle the dataframe
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:validation_end], df[validation_end:]
    

train_df, valid_df, test_df = train_valid_test_split(balanced_df, 0.7, 0.1)

print(f"Total size: {len(train_df)+len(valid_df)+len(test_df)}, Train size: {len(train_df)}, Valid size: {len(valid_df)}, Test size: {len(test_df)}")

train_df.to_csv("to_ignore/train.csv", index=None)
valid_df.to_csv("to_ignore/valid.csv", index=None)
test_df.to_csv("to_ignore/test.csv", index=None)

# In order to have the same length for every example, we choose to pad the texts to the length of the longest text in the dataset.
# The alternative is to truncate the texts to a fixed length/shortest text in the dataset, but this may result in loss of information.

class SpamDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer

        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        # find the longest text in the dataset
        if max_length is None:
            self.max_length = self._retrieve_longest_encoded_length()
        else:
            self.max_length = max_length
            # truncate text
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
            
        # pad all texts to the max length with the pad token (<|endoftext|>, encoded as 50256 in GPT-2)
        self.encoded_texts = [encoded_text + ([pad_token_id] * (self.max_length - len(encoded_text))) for encoded_text in self.encoded_texts]

    def _retrieve_longest_encoded_length(self):
        max_length = 0
        for text in self.data["Text"]:
            encoded = self.tokenizer.encode(text)
            max_length = max(max_length, len(encoded))
        return max_length
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded_text = self.encoded_texts[idx]
        label = self.data.iloc[idx]["Label"]
        return (
             torch.tensor(encoded_text, dtype=torch.long),
             torch.tensor(label, dtype=torch.long)
        )

tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset("to_ignore/train.csv", tokenizer)

# Truncation with train length optional, as the max length is 120, and our model can handle 1024 tokens as context length
val_dataset = SpamDataset("to_ignore/valid.csv", tokenizer, max_length=train_dataset.max_length)
test_dataset = SpamDataset("to_ignore/test.csv", tokenizer, max_length=train_dataset.max_length)

num_workers = os.cpu_count()
batch_size = 8
torch.manual_seed(123)

# 3) Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

print("Sanity check")
for input_batch, target_batch in train_loader:
    pass
print(input_batch.shape, target_batch.shape)
assert input_batch.shape[0] == batch_size
assert input_batch.shape[1] == train_dataset.max_length

for input_batch, target_batch in val_loader:
    pass
print(input_batch.shape, target_batch.shape)
assert input_batch.shape[0] == batch_size
assert input_batch.shape[1] == train_dataset.max_length

for input_batch, target_batch in test_loader:
    pass
print(input_batch.shape, target_batch.shape)
assert input_batch.shape[0] == batch_size
assert input_batch.shape[1] == train_dataset.max_length

print("Batches Overview")
print(f'{len(train_loader)} batches in train loader')
print(f'{len(val_loader)} batches in validation loader')
print(f'{len(test_loader)} batches in test loader')

# Stage 2: Model Setup
# 4) Initialize model
CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model = SmolGPTModel(BASE_CONFIG)

# 5) Load pretrained weights
load_weights(model, gpt2_huggingface)
model.eval()

# Test to ensure the weights are loaded correctly
# text_1 = "Every effort moves you"
# token_ids = generate(
#     model=model,
#     idx=text_to_token_ids(text_1, tokenizer),
#     max_new_tokens=15,
#     context_size=BASE_CONFIG["context_length"],
# )
# print(token_ids_to_text(token_ids, tokenizer))

# 6) Modify model for fine-tuning
# We substitute the last layer of the model with a linear layer with 2 output units for binary classification.
# First of all let's freeze all the layers
for params in model.parameters():
    params.requires_grad = False

torch.manual_seed(123)
NUM_CLASSES = 2

# Then we replace the last layer
# By default requires_grad is set to True 'cause we're replacing the layer
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["embed_dim"], 
                                 out_features=NUM_CLASSES)

# It's sufficient, but to yield better results, we can fine-tune the last layer, the last layer norm and the last transformer layer

for params in model.ln.parameters():
    params.requires_grad = True

for params in model.transformers_layers[-1].parameters():
    params.requires_grad = True

# Let's verify the new model
text = "May the force be with"
enc = tokenizer.encode(text)
enc = torch.tensor(enc).unsqueeze(0)
with torch.no_grad():
    out = model(enc)
print(f'Output: {out}; Shape: {out.shape}') # previously instead of 5 we would have 50256 as second dimension

# To fine-tune we are interested only on the last output token
print(f"Last token output: {out[:, -1, :]}")

# 7) Implement evaluation utilities
def calc_accuracy_loader(dataloader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :] # only the last output token
            predictions = torch.argmax(logits, dim=-1)
            num_examples += predictions.shape[0]
            correct_predictions += ((predictions == target_batch).sum().item())
        else:
            break
    return correct_predictions / num_examples

# torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# print(f'Training accuracy before fine-tuning: {calc_accuracy_loader(train_loader, model, device, num_batches=10)}%')
# print(f'Validation accuracy before fine-tuning: {calc_accuracy_loader(val_loader, model, device, num_batches=10)}%')
# print(f'Test accuracy before fine-tuning: {calc_accuracy_loader(test_loader, model, device, num_batches=10)}%')

# Since accuracy is not differentiable, let's use the cross-entropy as a loss function
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :] # last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(dataloader, model, device, num_batches=None):
    total_loss = 0
    if len(dataloader) == 0:
        return float("inf")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# Let's compute the initial loss before fine-tuning
# with torch.no_grad():
#     train_loss = calc_loss_loader(train_loader, model, device, num_batches=10)
#     val_loss = calc_loss_loader(val_loader, model, device, num_batches=10)
#     test_loss = calc_loss_loader(test_loader, model, device, num_batches=10)

# print(f'Training loss before fine-tuning: {train_loss:.3f}')
# print(f'Validation loss before fine-tuning: {val_loss:.3f}')
# print(f'Test loss before fine-tuning: {test_loss:.3f}')

# Stage 3: Model fine-tuning and usage
# 8) Fine-tune model
import math
from torch import nn

def evaluate_model(model, train_loader, val_loader, device, num_batches):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches)
    model.train()
    return train_loss, val_loss

# TODO: To be tested
def better_training_loop(model, train_loader, val_loader, optimizer, device, eval_freq, eval_iter, num_epochs, warmup_steps=20, initial_lr=3e-5, min_lr=1e-6):
    train_losses, val_losses, train_accs, val_accs, track_example_seen, track_lrs = [], [], [], [], [], []
    examples_seen, global_step = 0, -1
    peak_lr = optimizer.param_groups[0]["lr"] # get the initial learning rate
    lr_increment = (peak_lr - initial_lr) / warmup_steps
    total_training_steps = len(train_loader) * num_epochs
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1
            
            # Adjust the learning rate based on the current phase (warmup or cosine annealing)
            if global_step < warmup_steps:
                # Linear warmup
                lr = initial_lr + global_step * lr_increment  
            else:
                # Cosine annealing after warmup
                progress = ((global_step - warmup_steps) / 
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            
            # Update the learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            track_lrs.append(lr)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            
            # Gradient clipping: prevent the exploding gradient problem by scaling the gradients if they exceed a certain threshold
            if global_step > warmup_steps:
                # clip the gradients to a maximum norm of 1.0, where the norm is the sum of the squares of the gradients (L2 norm)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            examples_seen += input_batch.shape[0]
            
            # Optional evaluation of the model
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_example_seen.append(examples_seen)
                print(f'Epoch {epoch} Step {global_step:03d}: '
                      f'Train Loss: {train_loss:.3f}, '
                      f'Val Loss: {val_loss:.3f}')
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        train_accs.append(train_accuracy)
        
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        val_accs.append(val_accuracy)
        
        print(f'Training accuracy: {train_accuracy*100:.2f}% | Validation accuracy: {val_accuracy*100:.2f}%')
        
    return train_losses, val_losses, train_accs, val_accs, examples_seen

# import time
# start_time = time.time()
# torch.manual_seed(123)
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
# NUM_EPOCHS = 5

# train_losses, val_losses, train_accs, val_accs, examples_seen = better_training_loop(
#     model, train_loader, val_loader, optimizer, device, eval_freq=50, eval_iter=5, num_epochs=NUM_EPOCHS
# )

# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60
# print(f"Training complete in {execution_time_minutes:.2f} minutes")

# Overall the same as `train_model_simple` in chapter 5
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

# import time

# start_time = time.time()

# torch.manual_seed(123)

# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

# num_epochs = 5
# train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
#     model, train_loader, val_loader, optimizer, device,
#     num_epochs=num_epochs, eval_freq=50, eval_iter=5,
# )

# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60
# print(f"Training completed in {execution_time_minutes:.2f} minutes.")

import matplotlib.pyplot as plt

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="Loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.savefig(f"{label}-plot.pdf")
    plt.show()

# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

# plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

# # 9) Evaluate fine-tuned model
# print(f'Training accuracy after fine-tuning: {calc_accuracy_loader(train_loader, model, device)*100:.3f}%')
# print(f'Validation accuracy after fine-tuning: {calc_accuracy_loader(val_loader, model, device)*100:.3f}%')
# print(f'Test accuracy after fine-tuning: {calc_accuracy_loader(test_loader, model, device)*100:.3f}%')

# 10) Use model on new data
def classify_sms(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    
    encoded_text = tokenizer.encode(text)
    supported_context_length = model.position_embedding.weight.shape[1]
    
    # Truncates if too long
    encoded_text = encoded_text[:min(max_length, supported_context_length)]
    
    # Pad if too short
    encoded_text += [pad_token_id] * (max_length - len(encoded_text))
    
    # Add batch dimension
    encoded_text = torch.tensor(encoded_text).unsqueeze(0).to(device)
    
    # Get the model's prediction
    with torch.no_grad():
        logits = model(encoded_text)[:, -1, :]
    prediction = torch.argmax(logits, dim=-1).item()
    
    return "spam" if prediction == 1 else "not spam"

# text_1 = (
#     "You are a winner you have been specially"
#     " selected to receive $1000 cash or a $2000 award."
# )

# print(classify_sms(
#     text_1, model, tokenizer, device, max_length=train_dataset.max_length
# ))

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

# Save the fine-tuned model
# torch.save(model.state_dict(), "to_ignore/spam_classifier.pth")

# To load:
# model_state_dict = torch.load("to_ignore/spam_classifier.pth", map_location=device, weights_only=True)
# model.load_state_dict(model_state_dict)

# token_ids = classify_sms

# print(classify_sms(
#     text_2, model, tokenizer, device, max_length=train_dataset.max_length
# ))

# LoRa fine-tuning
# Let's create the layer that incorporates LoRa and linear layers

class LoRa(nn.Module):
    def __init__(self, rank, alpha, in_dim, out_dim):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)) # The same initialization used for Linear layers in Pytorch
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim)) # B is zeroed so it doesn't affect the original weights 
        self.alpha = alpha       
            
    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)
    
class LinearWithLoRa(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRa(rank, alpha, linear.in_features, linear.out_features)
            
    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        # Replace the linear layer with a LinearWithLoRa layer
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRa(module, rank, alpha))
        else:
            # Recursively apply the function to the child module
            replace_linear_with_lora(module, rank, alpha)

# Let's freeze the model
total_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters before freezing: {total_params:,}")

for paral in model.parameters():
    paral.requires_grad = False

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters after freezing: {total_params:,}")

# Let's replace the last layer with a LinearWithLoRa layer
# Alpha is usually half, double or equal to the rank
replace_linear_with_lora(model, rank=16, alpha=16)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters after substitution: {total_params:,}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import time

start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)