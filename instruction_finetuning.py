# Step for instruction finetuning are the same as of the classification fine-tuning, with some differences in the data preparation and model configuration.

# The dataset is composed by (instruction, response) pairs, where the instruction is the input and the response is the target. The model is trained to generate the response given the instruction.

# Download and loading the dataset provided by Raschka
import json
import os
import urllib.request
import random

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


file_path = "to_ignore/instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))
print(f'Example entry: {data[random.randint(0, len(data))]}')

# To process the dataset, we will use the Alpaca style
def format_input(entry): 
    instruction_text = {
        f"Below is an instruction that describes a task."
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    }
    
    input_text = (f"\n\n### Input:\n{entry['input']}" if entry["input"] else "")
    return instruction_text + input_text

# train-val-test split
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
val_data = data[train_portion + test_portion:]
test_data = data[train_portion:train_portion + test_portion]

print(f'"Training set length: {len(train_data)}')
print(f'Validation set length: {len(val_data)}')
print(f'Test set length: {len(test_data)}')

from torch.utils.data import Dataset, DataLoader
import torch
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        # Format the input
        self.data = [format_input(entry) for entry in self.data]
        response_text = [f"\n\n## Response:\n{entry["response"]}" for entry in self.data]
        # Tokenize the data
        self.encoded_texts = [tokenizer.encode(text+response) for (text, response) in zip(self.data, response_text)]
    
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)
# Now to prepare batches we need the following steps:
# First of all instead of pad using the longest sequence in the dataset, we pad according to the max length in the batch
# Second, we need to add the target to the batch, which is the input shifted by 1 token to the right
# Then, we substitute all the endoftext token with -100, so the crossentropy made in pytorch ignore the padding tokens, except for the first one, which we will use to train the llm to stop the generation of the response.
# Another step that we will not do is masking the instruction and the input in the target text, so the llm does not memorize the input, limiting overfitting. This is done by adding -100 to the target where the input is present.
# To do so we define a custom collate function
# The collate function is used by the DataLoader to prepare the batches. It receives a list of samples and returns a batch.
def custom_collate_function(batch, device="cpu", pad_token_id=50256, allowed_max_length=None, ignore_index=-100):
    max_length = max(len(item)+1 for item in batch) # +1 for the endoftext token
    inputs_list, target_list = [], []
    
    for item in batch:
        new_item = item.copy()
        # Add endoftext token
        new_item += [pad_token_id]
        
        # Pad sequences to max_length
        new_item += [pad_token_id] * (max_length - len(new_item))
        
        inputs = torch.tensor(new_item[:-1], dtype=torch.long) # Remove the last token 
        targets = torch.tensor(new_item[1:], dtype=torch.long) # Shift the input to the right by 1
        
        # Replace all the pad tokens with -100 except for the first one
        mask = (targets == pad_token_id).float()
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Truncates to the maximum sequence length if needed
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        
        inputs_list.append(inputs)
        target_list.append(targets)
        
    inputs_batch = torch.stack(inputs_list).to(device)
    target_batch = torch.stack(target_list).to(device)
    return inputs_batch, target_batch        
        
batch = [
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
    ]

inputs, targets = custom_collate_function(batch)
print(inputs, targets)