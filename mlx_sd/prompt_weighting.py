"""Prompt emphasis syntax for CLIP text encoding.

Implements the `(text:weight)` and `(text)` emphasis syntax that ComfyUI's own CLIPTextEncode
supports, so prompts written for standard ComfyUI behave the same way here. `[` and `]` are
left as ordinary characters, matching ComfyUI's own default parser.

Weighting is applied the same way ComfyUI's default ("comfy") parser does it: each weighted
token's *contextualized* output embedding is blended toward/away from its embedding in an
unweighted ("empty prompt") encoding, by the token's weight. This module only covers the
parsing/tokenization half of that (turning `text` into per-token ids + weights); the blend
itself happens where the CLIP model output is consumed.
"""

from typing import List, Tuple

_UPWEIGHT = 1.1

_ESCAPE_MAP = {
    "\\(": "\0LPAREN\0",
    "\\)": "\0RPAREN\0",
}
_UNESCAPE_MAP = {placeholder: literal[1:] for literal, placeholder in _ESCAPE_MAP.items()}


def _escape_literal_parens(text: str) -> str:
    for literal, placeholder in _ESCAPE_MAP.items():
        text = text.replace(literal, placeholder)
    return text


def _unescape_literal_parens(text: str) -> str:
    for placeholder, literal in _UNESCAPE_MAP.items():
        text = text.replace(placeholder, literal)
    return text


def _split_top_level_groups(text: str) -> List[str]:
    """Split into plain-text runs and well-nested `(...)` groups (kept whole, parens
    included). Unbalanced parens are left as plain text."""
    result = []
    current = ""
    depth = 0
    for ch in text:
        if ch == "(":
            if depth == 0 and current:
                result.append(current)
                current = ""
            current += ch
            depth += 1
        elif ch == ")":
            current += ch
            if depth > 0:
                depth -= 1
                if depth == 0:
                    result.append(current)
                    current = ""
        else:
            current += ch
    if current:
        result.append(current)
    return result


def _split_top_level_commas(text: str) -> List[str]:
    """Split on top-level commas, i.e. commas outside any nested `(...)` group."""
    segments = []
    current = ""
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth = max(depth - 1, 0)
            current += ch
        elif ch == "," and depth == 0:
            segments.append(current)
            current = ""
        else:
            current += ch
    segments.append(current)
    return segments


def _segment_explicit_weight(segment: str):
    """If `segment` is plain text ending in a bare `:weight` (not itself a nested
    `(...)` group), return `(text_without_weight, weight)`; otherwise `(segment, None)`."""
    trimmed = segment.strip()
    if len(trimmed) >= 2 and trimmed[0] == "(" and trimmed[-1] == ")":
        return segment, None  # nested group -- let recursion parse it as a whole
    colon = trimmed.rfind(":")
    if colon > 0:
        try:
            return trimmed[:colon], float(trimmed[colon + 1 :])
        except ValueError:
            pass
    return segment, None


def _token_weights(text: str, current_weight: float) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for item in _split_top_level_groups(text):
        is_group = len(item) >= 2 and item[0] == "(" and item[-1] == ")"
        if not is_group:
            out.append((item, current_weight))
            continue

        inner = item[1:-1]

        # A group holding a comma-separated list where every item carries its own
        # explicit `:weight` -- e.g. `(a:0.9, b:0.9, c:0.9)` -- is a list of
        # independently-weighted phrases, not one phrase ending in a single weight.
        # Require every segment to be explicit so this never fires on an ordinary
        # `(a, b, c:0.9)` grouped-upweight list, which keeps single-weight parity
        # with ComfyUI's own parser.
        segments = _split_top_level_commas(inner)
        parsed_segments = [_segment_explicit_weight(seg) for seg in segments]
        if len(segments) > 1 and all(w is not None for _text, w in parsed_segments):
            for idx, (seg_text, seg_weight) in enumerate(parsed_segments):
                out.extend(_token_weights(seg_text, seg_weight))
                if idx < len(parsed_segments) - 1:
                    out.append((", ", current_weight))
            continue

        weight = current_weight * _UPWEIGHT
        colon = inner.rfind(":")
        if colon > 0:
            candidate = inner[colon + 1 :]
            try:
                weight = float(candidate)
                inner = inner[:colon]
            except ValueError:
                pass
        out.extend(_token_weights(inner, weight))
    return out


def parse_prompt_weights(text: str) -> List[Tuple[str, float]]:
    """Parse emphasis syntax into a list of (chunk_text, weight) pairs, in order."""
    escaped = _escape_literal_parens(text)
    pairs = _token_weights(escaped, 1.0)
    return [(_unescape_literal_parens(chunk), weight) for chunk, weight in pairs]


def _flat_ids_and_weights(tokenizer, text: str) -> Tuple[List[int], List[float]]:
    """Tokenize `text` (honoring emphasis syntax) into a flat, unbounded list of token ids
    and per-token weights -- no bos/eos/padding, no length limit."""
    ids: List[int] = []
    weights: List[float] = []
    for chunk_text, weight in parse_prompt_weights(text):
        if not chunk_text:
            continue
        chunk_ids = tokenizer.encode(chunk_text, add_special_tokens=False)
        ids.extend(chunk_ids)
        weights.extend([weight] * len(chunk_ids))
    return ids, weights


def tokenize_with_weights(tokenizer, text: str, max_length: int = 77) -> Tuple[List[int], List[float]]:
    """Tokenize `text` honoring emphasis syntax.

    Returns `(token_ids, weights)`, both of length `max_length`, laid out the same way a plain
    `tokenizer(text, padding="max_length", max_length=max_length, truncation=True)` call would
    (bos, ..., eos, pad, pad, ...) but with a per-token weight instead of an implicit 1.0.
    Unweighted text produces weight 1.0 for every real token, so plain prompts are unaffected.

    Content beyond `max_length - 2` real tokens is silently truncated -- use
    `tokenize_with_weights_chunks` instead where long prompts must not lose content.
    """
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos

    ids, weights = _flat_ids_and_weights(tokenizer, text)

    body_len = max(max_length - 2, 0)
    ids = ids[:body_len]
    weights = weights[:body_len]

    out_ids = [bos] + ids + [eos]
    out_weights = [1.0] + weights + [1.0]

    pad_count = max_length - len(out_ids)
    if pad_count > 0:
        out_ids.extend([pad] * pad_count)
        out_weights.extend([1.0] * pad_count)

    return out_ids, out_weights


def tokenize_with_weights_chunks(tokenizer, text: str, chunk_size: int = 75) -> List[Tuple[List[int], List[float]]]:
    """Tokenize `text` honoring emphasis syntax, splitting into as many `chunk_size`-token
    windows as needed instead of truncating -- matching ComfyUI's handling of prompts longer
    than one CLIP window (CLIP's position embeddings are fixed at 77, so each chunk is encoded
    through CLIP separately and the resulting embeddings are concatenated afterward).

    Returns a list of `(token_ids, weights)` pairs, each of length `chunk_size + 2`
    (bos, ..., eos, pad...), laid out the same way `tokenize_with_weights` lays out its single
    chunk. A prompt that fits in one window returns a single-element list with output
    identical to `tokenize_with_weights(tokenizer, text, max_length=chunk_size + 2)`. An empty
    prompt also returns a single (all-pad) chunk, never an empty list.
    """
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos

    ids, weights = _flat_ids_and_weights(tokenizer, text)
    max_length = chunk_size + 2

    if not ids:
        return [([bos, eos] + [pad] * chunk_size, [1.0] * max_length)]

    chunks: List[Tuple[List[int], List[float]]] = []
    for start in range(0, len(ids), chunk_size):
        body_ids = ids[start : start + chunk_size]
        body_weights = weights[start : start + chunk_size]

        out_ids = [bos] + body_ids + [eos]
        out_weights = [1.0] + body_weights + [1.0]

        pad_count = max_length - len(out_ids)
        if pad_count > 0:
            out_ids.extend([pad] * pad_count)
            out_weights.extend([1.0] * pad_count)

        chunks.append((out_ids, out_weights))
    return chunks
