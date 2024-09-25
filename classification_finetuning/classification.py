import torch

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