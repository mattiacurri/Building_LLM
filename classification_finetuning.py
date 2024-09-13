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