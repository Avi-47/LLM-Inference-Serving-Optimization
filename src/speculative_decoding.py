import torch


def cache_length(past_key_values):
    return past_key_values.get_seq_length()


def truncate_kv_cache(past_key_values, keep_length):
    past_key_values.crop(keep_length)
    return past_key_values


def speculative_round(target_model, draft_model, tokenizer, target_kv, draft_kv, last_token, k=4):
    pre_round_length = cache_length(target_kv)

    draft_tokens = []
    current_input = last_token
    current_draft_kv = draft_kv

    for _ in range(k):
        with torch.no_grad():
            out = draft_model(input_ids=current_input, past_key_values=current_draft_kv, use_cache=True)
        current_draft_kv = out.past_key_values
        current_input = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        draft_tokens.append(current_input)

    draft_tensor = torch.cat([last_token] + draft_tokens, dim=1)
    with torch.no_grad():
        target_out = target_model(input_ids=draft_tensor, past_key_values=target_kv, use_cache=True)
    target_kv = target_out.past_key_values
    target_logits = target_out.logits

    accept_len = 0
    corrected_token = None
    for i in range(k):
        target_pred = torch.argmax(target_logits[:, i, :], dim=-1, keepdim=True)
        if target_pred.item() == draft_tokens[i].item():
            accept_len += 1
        else:
            corrected_token = target_pred
            break

    if corrected_token is not None:
        new_last_token = corrected_token
        keep_length = pre_round_length + 1 + accept_len
        target_kv = truncate_kv_cache(target_kv, keep_length)
        draft_kv = truncate_kv_cache(current_draft_kv, keep_length)
    else:
        new_last_token = torch.argmax(target_logits[:, k, :], dim=-1, keepdim=True)
        with torch.no_grad():
            extra = draft_model(input_ids=draft_tokens[k - 1], past_key_values=current_draft_kv, use_cache=True)
        draft_kv = extra.past_key_values

    accepted_tokens = [t.item() for t in draft_tokens[:accept_len]] + [new_last_token.item()]
    return accepted_tokens, new_last_token, target_kv, draft_kv


def speculative_generate(target_model, draft_model, tokenizer, prompt, k=4, max_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        target_out = target_model(input_ids, use_cache=True)
        draft_out = draft_model(input_ids, use_cache=True)

    target_kv = target_out.past_key_values
    draft_kv = draft_out.past_key_values
    last_token = torch.argmax(target_out.logits[:, -1, :], dim=-1, keepdim=True)

    tokens_yielded = 1
    yield last_token.item()
    if last_token.item() == tokenizer.eos_token_id:
        return

    while tokens_yielded < max_tokens:
        round_k = min(k, max_tokens - tokens_yielded)
        accepted_tokens, last_token, target_kv, draft_kv = speculative_round(
            target_model, draft_model, tokenizer, target_kv, draft_kv, last_token, k=round_k
        )
        for t in accepted_tokens:
            if tokens_yielded >= max_tokens:
                return
            yield t
            tokens_yielded += 1
            if t == tokenizer.eos_token_id:
                return
