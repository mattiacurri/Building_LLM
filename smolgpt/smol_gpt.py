import torch
from SmolGPT import LayerNorm, TransformerBlock, SmolGPTModel, SMOLGPT_CONFIG_124M

# Test of LayerNorm
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
ln = LayerNorm(embed_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(-1, keepdim=True)
var = out_ln.var(-1, keepdim=True, unbiased=False)
print(f'Mean: {mean}')
print(f'Variance: {var}')

# Test of TransformerBlock
torch.manual_seed(123)
x = torch.randn(2, 4, 768)
block = TransformerBlock(SMOLGPT_CONFIG_124M)
out_block = block(x)
print(f'Input shape: {x.shape}')
print(f'Output shape: {out_block.shape}')

# Now let's initialize the 124 mln parameter GPT model
torch.manual_seed(123)

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

SmolGPT = SmolGPTModel(SMOLGPT_CONFIG_124M)

out = SmolGPT(batch)
print(f'Input batch: {batch}')
print(f'Output shape: {out.shape}')
print(out)

total_params = sum(p.numel() for p in SmolGPT.parameters())
print(f'Total parameters in SmolGPT: {total_params:,}')

# Weight tying: the embedding layer and the final linear layer share the same weights, that's the reason we have 163 mln parameters instead of 124 mln
# Better training and model performance according to Raschka
total_params_gpt2 =  total_params - sum(p.numel() for p in SmolGPT.out_head.parameters())
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")


mha_params = sum(p.numel() for x in SmolGPT.transformers_layers for p in x.attention.parameters()) / 12
ff_params = sum(p.numel() for x in SmolGPT.transformers_layers for p in x.feed_forward.parameters()) / 12
print(f'Parameters in a single MHA: {mha_params:,}')
print(f'Parameters in a single FF: {ff_params:,}')

# Memory requirements
total_size_bytes = total_params * 4 # Assuming each parameter is a 32-bit float
total_size_mb = total_size_bytes / (1024 * 1024)
print(f'Total memory requirements: {total_size_mb:.2f} MB')

# Let's move on to how to generate text with SmolGPT
# For now we will generate text based on the most probable token at each step
# Further improvements can be made by sampling from the distribution of the logits (temperature, top-k, top-p sampling, ...)
from SmolGPT import generate_text_simple

start_context = "Hello, I am"

encoded = tokenizer.encode(start_context)
print("encoded:", encoded)

encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

SmolGPT.eval() # disable dropout

out = generate_text_simple(
    model=SmolGPT,
    idx=encoded_tensor, 
    max_new_tokens=6, 
    context_size=SMOLGPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)