import copy
import gc
import threading
from collections import OrderedDict

import torch

MIN_SHARED_PREFIX_LEN = 3
MODEL_CACHE_VERSION = "qwen2.5-1.5b_target_v1"
DRAFT_CACHE_VERSION = "qwen2.5-0.5b_draft_v1"
MAX_PREFIX_CACHE_ENTRIES = 128


class PrefixCache:
    """Bounded, version-aware cache for immutable prefix snapshots."""

    def __init__(self, max_entries=MAX_PREFIX_CACHE_ENTRIES):
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self):
        with self._lock:
            return len(self._entries)

    def __contains__(self, prefix_ids):
        with self._lock:
            return tuple(prefix_ids) in self._entries

    def __getitem__(self, prefix_ids):
        with self._lock:
            entry = self._entries[tuple(prefix_ids)]
            self._entries.move_to_end(tuple(prefix_ids))
            return entry

    def __setitem__(self, prefix_ids, entry):
        self.put(prefix_ids, entry)

    def items(self):
        with self._lock:
            return list(self._entries.items())

    def clear(self):
        self.invalidate_prefix_cache(clear=True)

    def get(self, prefix_ids, default=None, *, target_version=None, draft_version=None):
        key = tuple(prefix_ids)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return default
            if (target_version is not None and entry.get("target_version") != target_version) or \
                    (draft_version is not None and entry.get("draft_version") != draft_version):
                self._remove_locked(key)
                self.misses += 1
                return default
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def metrics(self):
        with self._lock:
            total = self.hits + self.misses
            return {
                "cache_hits": self.hits,
                "cache_misses": self.misses,
                "cache_hit_rate": self.hits / total if total else 0.0,
                "evictions": self.evictions,
                "entries": len(self._entries),
            }

    def reset_metrics(self):
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0

    def put(self, prefix_ids, entry):
        key = tuple(prefix_ids)
        with self._lock:
            old_entry = self._entries.pop(key, None)
            del old_entry
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                _, evicted = self._entries.popitem(last=False)
                del evicted
                self.evictions += 1
        gc.collect()

    def _remove_locked(self, key):
        entry = self._entries.pop(key, None)
        del entry

    def invalidate_prefix_cache(self, prefix_ids=None, target_version=None,
                                draft_version=None, clear=False):
        with self._lock:
            if clear:
                removed = list(self._entries.values())
                self._entries.clear()
            elif prefix_ids is not None:
                key = tuple(prefix_ids)
                removed = [self._entries.pop(key)] if key in self._entries else []
            else:
                removed = []
                for key, entry in list(self._entries.items()):
                    if ((target_version is None or entry.get("target_version") == target_version) and
                            (draft_version is None or entry.get("draft_version") == draft_version)):
                        removed.append(self._entries.pop(key))
        del removed
        gc.collect()


prefix_cache = PrefixCache()
prefix_cache_pool = prefix_cache


def shared_prefix_len(tokens_a, tokens_b):
    length = 0
    for a, b in zip(tokens_a, tokens_b):
        if a != b:
            break
        length += 1
    return length


def find_cached_prefix(token_ids, target_version=MODEL_CACHE_VERSION,
                       draft_version=None):
    best = None
    for cached_ids, _ in prefix_cache_pool.items():
        n = len(cached_ids)
        if (len(token_ids) >= n and tuple(token_ids[:n]) == cached_ids and
                prefix_cache_pool.get(cached_ids, target_version=target_version,
                                      draft_version=draft_version) is not None):
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
        "first_token": next_token,
        "target_version": MODEL_CACHE_VERSION,
    }


def invalidate_prefix_cache(prefix_ids=None, target_version=None, draft_version=None,
                            clear=False):
    """Remove one prefix, entries matching versions, or the complete cache."""
    prefix_cache.invalidate_prefix_cache(prefix_ids, target_version, draft_version, clear)


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
