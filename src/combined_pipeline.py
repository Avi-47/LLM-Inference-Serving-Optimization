import copy
import torch

from prefix_caching import find_cached_prefix, shared_prefix_len, prefix_cache_pool, MIN_SHARED_PREFIX_LEN
from speculative_decoding import speculative_round

spec_stats = {"rounds": 0, "accepted": 0, "drafted": 0}


def cache_prefix_both(target_model, draft_model, tokenizer, token_ids):
    input_ids = torch.tensor([token_ids]).to("cuda")
    with torch.no_grad():
        target_out = target_model(input_ids, use_cache=True)
        draft_out = draft_model(input_ids, use_cache=True)
    first_token = torch.argmax(target_out.logits[:, -1, :], dim=-1, keepdim=True)
    prefix_cache_pool[tuple(token_ids)] = {
        "target_kv": target_out.past_key_values,
        "draft_kv": draft_out.past_key_values,
        "first_token": first_token
    }


def prefill_suffix_both(target_model, draft_model, tokenizer, cached_entry, suffix_ids):
    target_kv = copy.deepcopy(cached_entry["target_kv"])
    draft_kv = copy.deepcopy(cached_entry["draft_kv"])

    if suffix_ids:
        suffix_tensor = torch.tensor([suffix_ids]).to("cuda")
        with torch.no_grad():
            target_out = target_model(input_ids=suffix_tensor, past_key_values=target_kv, use_cache=True)
            draft_out = draft_model(input_ids=suffix_tensor, past_key_values=draft_kv, use_cache=True)
        target_kv = target_out.past_key_values
        draft_kv = draft_out.past_key_values
        last_token = torch.argmax(target_out.logits[:, -1, :], dim=-1, keepdim=True)
    else:
        last_token = cached_entry["first_token"]

    return target_kv, draft_kv, last_token


def combined_generate(target_model, draft_model, tokenizer, prompt, waiting_prompts,
                       use_prefix_cache=True, use_speculative=True, k=4, max_tokens=20):
    token_ids = tokenizer(prompt)["input_ids"]
    matched_prefix = None

    if use_prefix_cache:
        matched_prefix = find_cached_prefix(token_ids)
        if matched_prefix is None:
            for other_prompt in waiting_prompts:
                other_ids = tokenizer(other_prompt)["input_ids"]
                shared_len = shared_prefix_len(token_ids, other_ids)
                if shared_len >= MIN_SHARED_PREFIX_LEN:
                    cache_prefix_both(target_model, draft_model, tokenizer, token_ids[:shared_len])
                    matched_prefix = tuple(token_ids[:shared_len])
                    break

    if matched_prefix:
        suffix_ids = token_ids[len(matched_prefix):]
        cached_entry = prefix_cache_pool[matched_prefix]
        target_kv, draft_kv, last_token = prefill_suffix_both(target_model, draft_model, tokenizer, cached_entry, suffix_ids)
    else:
        input_ids = torch.tensor([token_ids]).to("cuda")
        with torch.no_grad():
            target_out = target_model(input_ids, use_cache=True)
        target_kv = target_out.past_key_values
        last_token = torch.argmax(target_out.logits[:, -1, :], dim=-1, keepdim=True)
        draft_kv = None
        if use_speculative:
            with torch.no_grad():
                draft_out = draft_model(input_ids, use_cache=True)
            draft_kv = draft_out.past_key_values

    tokens_yielded = 1
    yield last_token.item()
    if last_token.item() == tokenizer.eos_token_id:
        return

    while tokens_yielded < max_tokens:
        if use_speculative:
            round_k = min(k, max_tokens - tokens_yielded)
            accepted_tokens, last_token, target_kv, draft_kv = speculative_round(
                target_model, draft_model, tokenizer, target_kv, draft_kv, last_token, k=round_k
            )
            spec_stats["rounds"] += 1
            spec_stats["accepted"] += len(accepted_tokens)
            spec_stats["drafted"] += round_k
        else:
            with torch.no_grad():
                out = target_model(input_ids=last_token, past_key_values=target_kv, use_cache=True)
            target_kv = out.past_key_values
            last_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            accepted_tokens = [last_token.item()]

        for t in accepted_tokens:
            if tokens_yielded >= max_tokens:
                return
            yield t
            tokens_yielded += 1
            if t == tokenizer.eos_token_id:
                return
