# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path
import sys

import tiktoken
import torch
import chainlit

import os
import sys

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification_finetuning.classification import classify_sms
from smolgpt.SmolGPT import SmolGPTModel, generate, text_to_token_ids, token_ids_to_text
from smolgpt.gpt2 import GPT2_BASE_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_and_tokenizer(kind_of_model="classification"):
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "embed_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")
    
    if kind_of_model == "classification":
        model_path = Path("..") / "to_ignore" / "spam_classifier.pth"
        if not model_path.exists():
            print(
                f"Could not find the {model_path} file."
            )
            sys.exit()
    elif kind_of_model == "instruction":
        model_path = Path("..") / "to_ignore" / "gpt2_small_finetuned.pth"
        if not model_path.exists():
            print(
                f"Could not find the {model_path} file."
            )
            sys.exit()
        

    # Instantiate model
    model = SmolGPTModel(GPT_CONFIG_124M)

    if kind_of_model == "classification":
        num_classes = 2
        model.out_head = torch.nn.Linear(in_features=GPT_CONFIG_124M["embed_dim"], out_features=num_classes)

    # Then load model weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return tokenizer, model


# Obtain the necessary tokenizer and model files for the chainlit function below
KIND_OF_MODEL = "instruction"
tokenizer, model = get_model_and_tokenizer(KIND_OF_MODEL)


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    user_input = message.content
    response = classify_sms(user_input, model, tokenizer, device, max_length=120) if KIND_OF_MODEL == "classification" \
                else generate(model, 
                              text_to_token_ids(user_input, tokenizer).to(device),
                              max_new_tokens=1024, 
                              context_size=GPT2_BASE_CONFIG["context_length"], 
                              eos_id=50256)
    if KIND_OF_MODEL == "classification":
        await chainlit.Message(
            content=f"{response}",  # This returns the model response to the interface
        ).send()
    else:
        await chainlit.Message(
            content=f"{token_ids_to_text(response, tokenizer)}",  # This returns the model response to the interface
        ).send()