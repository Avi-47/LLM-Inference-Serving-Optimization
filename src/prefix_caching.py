import copy
import torch

MIN_SHARED_PREFIX_LEN = 3
prefix_cache_pool = {}


def shared_prefix_len(tokens_a, tokens_b):
    length = 0
    for a, b in zip(tokens_a, tokens_b):
        if a != b:
            break
        length += 1
    return length


def find_cached_prefix(token_ids):
    best = None
    for cached_ids in prefix_cache_pool:
        n = len(cached_ids)
        if len(token_ids) >= n and tuple(token_ids[:n]) == cached_ids:
            if best is None or n > len(best):
                best = cached_ids
    return best


def cache_prefix(model, tokenizer, token_ids):
    input_ids = torch.tensor([token_ids]).to("cuda")
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    prefix_cache_pool[tuple(token_ids)] = {
        "kv": out.past_key_values,
        "first_token": next_token
    }


def generate_tokens_with_prefix_ids(model, tokenizer, cached_entry, suffix_ids, max_tokens=20):
    past_key_values = copy.deepcopy(cached_entry["kv"])

    if suffix_ids:
        input_ids = torch.tensor([suffix_ids]).to("cuda")
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    else:
        next_token = cached_entry["first_token"]

    yield next_token.item()

    for _ in range(max_tokens - 1):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        yield next_token.item()


def resolve_prompt(model, tokenizer, prompt, waiting_prompts, max_tokens=20):
    from decode import generate_tokens

    token_ids = tokenizer(prompt)["input_ids"]
    matched_prefix = find_cached_prefix(token_ids)

    if matched_prefix is None:
        for other_prompt in waiting_prompts:
            other_ids = tokenizer(other_prompt)["input_ids"]
            shared_len = shared_prefix_len(token_ids, other_ids)
            if shared_len >= MIN_SHARED_PREFIX_LEN:
                cache_prefix(model, tokenizer, token_ids[:shared_len])
                matched_prefix = tuple(token_ids[:shared_len])
                break

    if matched_prefix:
        suffix_ids = token_ids[len(matched_prefix):]
        cached_entry = prefix_cache_pool[matched_prefix]
        return generate_tokens_with_prefix_ids(model, tokenizer, cached_entry, suffix_ids, max_tokens)
    else:
        return generate_tokens(model, tokenizer, prompt, max_tokens)
