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