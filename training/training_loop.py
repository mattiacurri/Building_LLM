import torch
import torch.nn as nn
import math
import os
import sys

# plot stuff
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# add in sys.path the path of the folder containing the module
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smolgpt.SmolGPT import generate_and_print_sample 
    
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

# Let's create a training loop
def evaluate_model(model, train_loader, val_loader, device, num_batches):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches)
    model.train()
    return train_loss, val_loss

def training_loop_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer=None):
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
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_token_seen

# Let's plot the training and validation loss
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, xlabel, file_name, label="Loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label=f"Training {label}")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel(xlabel)

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(file_name)
    plt.show()

# Now a better training loop
def better_training_loop(model, train_loader, val_loader, optimizer, device, eval_freq, eval_iter, start_context, num_epochs, warmup_steps=20, initial_lr=3e-5, min_lr=1e-6):
    train_losses, val_losses, track_token_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1
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
            tokens_seen += input_batch.numel()
            
            # Optional evaluation of the model
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_freq)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)
                print(f'Epoch: {epoch}, Global Step: {global_step}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')
        generate_and_print_sample(model, train_loader.dataset.tokenizer, device, start_context)
        
    return train_losses, val_losses, track_token_seen, track_lrs

