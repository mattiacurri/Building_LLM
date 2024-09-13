import torch
import numpy as np
from training import generate, text_to_token_ids, tokenizer, token_ids_to_text
import tiktoken
from transformers import GPT2Model
from smol_gpt import SmolGPTModel

model_names = {
    "gpt2-small (124M)": "openai-community/gpt2",
    "gpt2-medium (355M)": "openai-community/gpt2-medium",
    "gpt2-large (774M)": "openai-community/gpt2-large",
    "gpt2-xl (1558M)": "openai-community/gpt2-xl"
}

CHOOSE_MODEL = "gpt2-small (124M)"

gpt2_huggingface = GPT2Model.from_pretrained(model_names[CHOOSE_MODEL], cache_dir="to_ignore/checkpoints")
gpt2_huggingface.eval()
print(gpt2_huggingface)

BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": True        # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"embed_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"embed_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"embed_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"embed_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

def assign_check(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())

def load_weights(gpt, gpt_hf):

    d = gpt_hf.state_dict()

    gpt.position_embedding.weight = assign_check(gpt.position_embedding.weight, d["wpe.weight"])
    gpt.token_embedding.weight = assign_check(gpt.token_embedding.weight, d["wte.weight"])
    
    for b in range(BASE_CONFIG["n_layers"]):
        q_w, k_w, v_w = np.split(d[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.transformers_layers[b].attention.W_query.weight = assign_check(gpt.transformers_layers[b].attention.W_query.weight, q_w.T)
        gpt.transformers_layers[b].attention.W_key.weight = assign_check(gpt.transformers_layers[b].attention.W_key.weight, k_w.T)
        gpt.transformers_layers[b].attention.W_value.weight = assign_check(gpt.transformers_layers[b].attention.W_value.weight, v_w.T)
    
        q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.transformers_layers[b].attention.W_query.bias = assign_check(gpt.transformers_layers[b].attention.W_query.bias, q_b)
        gpt.transformers_layers[b].attention.W_key.bias = assign_check(gpt.transformers_layers[b].attention.W_key.bias, k_b)
        gpt.transformers_layers[b].attention.W_value.bias = assign_check(gpt.transformers_layers[b].attention.W_value.bias, v_b)
    
    
        gpt.transformers_layers[b].attention.out_proj.weight = assign_check(gpt.transformers_layers[b].attention.out_proj.weight, d[f"h.{b}.attn.c_proj.weight"].T)
        gpt.transformers_layers[b].attention.out_proj.bias = assign_check(gpt.transformers_layers[b].attention.out_proj.bias, d[f"h.{b}.attn.c_proj.bias"])
    
        gpt.transformers_layers[b].feed_forward.layers[0].weight = assign_check(gpt.transformers_layers[b].feed_forward.layers[0].weight, d[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.transformers_layers[b].feed_forward.layers[0].bias = assign_check(gpt.transformers_layers[b].feed_forward.layers[0].bias, d[f"h.{b}.mlp.c_fc.bias"])
        gpt.transformers_layers[b].feed_forward.layers[2].weight = assign_check(gpt.transformers_layers[b].feed_forward.layers[2].weight, d[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.transformers_layers[b].feed_forward.layers[2].bias = assign_check(gpt.transformers_layers[b].feed_forward.layers[2].bias, d[f"h.{b}.mlp.c_proj.bias"])
    
        gpt.transformers_layers[b].layer_norm_1.scale = assign_check(gpt.transformers_layers[b].layer_norm_1.scale, d[f"h.{b}.ln_1.weight"])
        gpt.transformers_layers[b].layer_norm_1.shift = assign_check(gpt.transformers_layers[b].layer_norm_1.shift, d[f"h.{b}.ln_1.bias"])
        gpt.transformers_layers[b].layer_norm_2.scale = assign_check(gpt.transformers_layers[b].layer_norm_2.scale, d[f"h.{b}.ln_2.weight"])
        gpt.transformers_layers[b].layer_norm_2.shift = assign_check(gpt.transformers_layers[b].layer_norm_2.shift, d[f"h.{b}.ln_2.bias"])
    
        gpt.ln.scale = assign_check(gpt.ln.scale, d[f"ln_f.weight"])
        gpt.ln.shift = assign_check(gpt.ln.shift, d[f"ln_f.bias"])
        gpt.out_head.weight = assign_check(gpt.out_head.weight, d["wte.weight"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt2 = SmolGPTModel(BASE_CONFIG)
load_weights(gpt2, gpt2_huggingface)

torch.manual_seed(123)
token_ids = generate(
    model=gpt2.to(device),
    idx=text_to_token_ids("Every effort moves", tokenizer).to(device),
    max_new_tokens=30,
    context_size=BASE_CONFIG["context_length"],
    top_k=1,
    temperature=1.0
)

print(token_ids_to_text(token_ids, tokenizer))