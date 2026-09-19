import torch


def manual_generate(model, tokenizer, prompt, max_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        out = model(input_ids, use_cache=True)

    past_key_values = out.past_key_values

    next_token_logits = out.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=1, keepdim=True)

    generated = [next_token.item()]

    for _ in range(max_tokens - 1):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)

        past_key_values = out.past_key_values
        next_token_logits = out.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_tokens(model, tokenizer, prompt, max_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(inputs["input_ids"], use_cache=True)
    past_key_values = out.past_key_values
    next_token_logits = out.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    yield next_token.item()

    for _ in range(max_tokens - 1):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        next_token_logits = out.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        yield next_token.item()
