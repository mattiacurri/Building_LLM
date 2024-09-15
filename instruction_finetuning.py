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
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    
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
        input_text = [format_input(entry) for entry in data]
        response_text = [f"\n\n## Response:\n{entry["output"]}" for entry in data]
        # Tokenize the data
        self.encoded_texts = [tokenizer.encode(text+response) for (text, response) in zip(input_text, response_text)]
    
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

# Now we can create the dataloaders
torch.manual_seed(123)
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"

import os
from functools import partial
# We use a partial function to pass the device and the maximum length to the collate function, so we don't need to pass them every time we call the collate function
customized_collate_function = partial(custom_collate_function, 
                                      device=device, 
                                      allowed_max_length=1024 # gpt2 max length is 1024
                                      )
NUM_WORKERS = os.cpu_count() if device == "cpu" else 0
BATCH_SIZE = 8

train_loader = DataLoader(
    InstructionDataset(train_data, tokenizer),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True,
    collate_fn=customized_collate_function
)

val_loader = DataLoader(
    InstructionDataset(val_data, tokenizer),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS,
    collate_fn=customized_collate_function
)

test_loader = DataLoader(
    InstructionDataset(test_data, tokenizer),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS,
    collate_fn=customized_collate_function
)

print("Sanity check")
for x, y in train_loader:
    assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1], f"Train loader: {x.shape} != {y.shape}"
for x, y in val_loader:
    assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1], f"Val loader: {x.shape} != {y.shape}"
for x, y in test_loader:
    assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1], f"Test loader: {x.shape} != {y.shape}"
print("All checks passed")

# Let's load gpt2-small
from gpt2 import BASE_CONFIG, load_weights, model_configs, model_names
from smol_gpt import SmolGPTModel
from transformers import GPT2Model
from training import generate, token_ids_to_text, text_to_token_ids

CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

gpt2_huggingface = GPT2Model.from_pretrained(model_names[CHOOSE_MODEL], cache_dir="to_ignore/checkpoints")
gpt2_huggingface.eval()
print(gpt2_huggingface)

gpt2 = SmolGPTModel(BASE_CONFIG)
load_weights(gpt2, gpt2_huggingface)
gpt2.eval()

# Baseline without fine-tuning
torch.manual_seed(123)
input_text = format_input(val_data[0])
print(f"Input text: {input_text}")
print()
token_ids = generate(gpt2, 
                     text_to_token_ids(input_text, tokenizer), 
                     max_new_tokens=35, 
                     context_size=BASE_CONFIG["context_length"], 
                     eos_id=50256)
# generate repeats the input, let's remove it
print(f"Response: {token_ids_to_text(token_ids, tokenizer)[len(input_text):].strip()}")

# Now let's fine-tune
# We can use the functions used for the training 🤗
from training import calc_loss_loader, training_loop_simple

# Let's calculate the initial loss
gpt2.to(device)
torch.manual_seed(123)
with torch.no_grad():
    initial_train_loss = calc_loss_loader(train_loader, gpt2, device, num_batches=5)
    initial_val_loss = calc_loss_loader(val_loader, gpt2, device, num_batches=5)

print(f"Initial train loss: {initial_train_loss}")
print(f"Initial val loss: {initial_val_loss}")

# Training
import time
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.Adam(gpt2.parameters(), lr=5e-5, weight_decay=0.01)
NUM_EPOCHS = 3

# Unfortunately gpt2-small is very bad but I have only 4 GB of VRAM locally
# train_losses, val_losses, tokens_seen = training_loop_simple(
#     gpt2, 
#     train_loader, 
#     val_loader, 
#     optimizer, 
#     device, 
#     num_epochs=NUM_EPOCHS, 
#     eval_freq=5, 
#     eval_iter=5, 
#     start_context=format_input(val_data[0]),
#     tokenizer=tokenizer)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

file_name = "to_ignore/gpt2_small_finetuned.pth"
# torch.save(gpt2.state_dict(), file_name)
# print(f"Model saved to {file_name}")

# from training import plot_losses

# epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# load the model
gpt2.load_state_dict(torch.load(file_name))
gpt2.eval()

# Evaluation part
# First of all we need to extract the response of the model on our test set
torch.manual_seed(123)

NO_SAMPLE = 3

for entry in test_data[:NO_SAMPLE]:
    input_text = format_input(entry)
    token_ids = generate(
        gpt2, 
        text_to_token_ids(input_text, tokenizer).to(device), 
        max_new_tokens=256, 
        context_size=BASE_CONFIG["context_length"], 
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    
    response_text = generated_text[len(input_text):].replace("## Response:", "").strip()
    
    print(input_text)
    print(f"\nCorrect response:\n >> {entry['output']}")
    print(f"\nModel response:\n >> {response_text}")
    print("\n" + "-"*100 + "\n")

# The evaluation is different from the classification fine-tuning, as we are not interested in the accuracy of the model, but in the quality of the generated text. 
# There three main approches to evaluate the quality of the generated text:
# 1. Human evaluation: we can ask human evaluators to rate the quality of the generated text, like in the LLM Arena
# 2. MMLU: Short-answer and multiple-choice benchmarks can be used to test the general knowledge of the model
# 3. Automated conversational benchmarks, where another LLM is used to evaluate the responses
# We will use the third approach, as we are not interested only on short answer and multiple choice and human evaluation of 1100 responses is time-consuming

# First of all, we add the responses generated by the model to the test set dictionary and save it
from tqdm import tqdm
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    token_ids = generate(
        gpt2, 
        text_to_token_ids(input_text, tokenizer).to(device), 
        max_new_tokens=256, 
        top_k=5,
        context_size=BASE_CONFIG["context_length"], 
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    
    response_text = generated_text[len(input_text):].replace("## Response:", "").strip()
    
    test_data[i]["model_response"] = response_text

# Save the test set with the model responses
with open("to_ignore/test_data_with_model_responses.json", "w") as file:
    json.dump(test_data, file, indent=4) # indent for pretty printing
    
# Now we check if ollama is running, we will use it to run an LLM to evaluate the responses
import psutil

def check_ollama(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_ollama("ollama")

if not ollama_running:
    raise RuntimeError("Ollama is not running. Please start it before running the evaluation.")

print(f"Ollama running: {check_ollama("ollama")}")

# In alternative we can communicate with the ollama server using the requests library
import requests

def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data
    
print(query_model(prompt="What do Llamas eat?", model="llama3"))

# Now let's evaluate the responses
    
from tqdm import tqdm
def evaluate_responses(test_data, judge="llama3"):
    scores = []
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"give a score to the model response `{entry['model_response']}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
        score = query_model(prompt=prompt, model=judge)
        print(f'Input: {prompt}')
        print(f'Output: {entry["output"]}')
        print(f"Model response: {entry['model_response']}")
        print(f"Score assigned: {score}")
        print("-"*100)
        scores.append(int(score))
    return scores

scores = evaluate_responses(test_data)
print(f"Average score: {sum(scores) / len(scores):.2f}")