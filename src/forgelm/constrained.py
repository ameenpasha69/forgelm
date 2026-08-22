"""Constrained decoding: make an illegal output unrepresentable.

Motivation from v1 evidence: 18 of 86 LoRA outputs contained an enum value that
does not exist -- `"dns"`, `"internet"`, `"display"`, `"audio"` -- almost always
a noun lifted out of the ticket instead of mapped onto the closed enum. That is
a *decoding* failure as much as a model failure: nothing stopped the model
emitting those tokens.

Approach
--------
The target language is tiny and completely fixed. Every legal output is

    {"category": "<CAT>", "priority": "<PRI>", "affected_service": "<SVC>",
     "is_security_incident": <BOOL>, "users_affected": <INT>}

with the literal punctuation exactly as `schema.canonical_json` emits it. So
rather than a general JSON grammar we implement an exact prefix automaton over
that one template, and mask every token that would take the string outside the
language.

This guarantees, by construction: exactly one JSON object, exactly the five
permitted keys in that order, correct value types, enum values only from the
permitted sets, no additional fields, no markdown fence and no surrounding
prose.

Fairness
--------
The same mechanism is applied to base-model and adapted-model conditions. The
quantity of interest is the difference *between two constrained conditions*,
never constrained-adapted against unconstrained-base -- that would credit the
model with what the decoder did.
"""

from __future__ import annotations

from typing import Any, Iterable

from .schema import AFFECTED_SERVICES, CATEGORIES, MAX_USERS_AFFECTED, PRIORITIES

# The literal segments, alternating with the five value slots. These must match
# `schema.canonical_json` exactly; a test asserts that they do.
LITERALS: tuple[str, ...] = (
    '{"category": "',
    '", "priority": "',
    '", "affected_service": "',
    '", "is_security_incident": ',
    ', "users_affected": ',
    "}",
)

BOOLEANS: tuple[str, ...] = ("true", "false")

# Slot i sits between LITERALS[i] and LITERALS[i + 1].
ENUM_SLOTS: tuple[tuple[str, ...], ...] = (
    CATEGORIES,
    PRIORITIES,
    AFFECTED_SERVICES,
    BOOLEANS,
)
INT_SLOT_INDEX = 4          # users_affected
N_SLOTS = 5

_MAX_INT_DIGITS = len(str(MAX_USERS_AFFECTED))


def _int_prefix_state(text: str) -> tuple[bool, bool]:
    """(is_valid_prefix, could_stop_here) for the users_affected slot."""
    if not text:
        return True, False
    if not text.isdigit():
        return False, False
    if text[0] == "0":                       # no leading zeros, and >= 1
        return False, False
    if len(text) > _MAX_INT_DIGITS:
        return False, False
    value = int(text)
    if value > MAX_USERS_AFFECTED:
        return False, False
    # A longer number may still be reachable, e.g. "5" -> "50".
    return True, value >= 1


def analyse(text: str) -> tuple[bool, bool]:
    """Classify `text` against the output language.

    Returns (is_valid_prefix, is_complete). `is_valid_prefix` means the string
    could still be extended into a legal output; `is_complete` means it already
    is one.
    """
    pos = 0
    n = len(text)

    for index in range(len(LITERALS)):
        literal = LITERALS[index]
        available = n - pos
        if available < len(literal):
            # Partially through this literal: must match what we have.
            return text[pos:] == literal[:available], False
        if text[pos:pos + len(literal)] != literal:
            return False, False
        pos += len(literal)

        if index == len(LITERALS) - 1:
            # Consumed the final "}"; anything after it is illegal.
            return pos == n, pos == n

        if index == INT_SLOT_INDEX:
            # Integer slot runs until the closing brace.
            remainder = text[pos:]
            digits = remainder
            closing = ""
            if "}" in remainder:
                cut = remainder.index("}")
                digits, closing = remainder[:cut], remainder[cut:]
            valid, can_stop = _int_prefix_state(digits)
            if not valid:
                return False, False
            if not closing:
                return True, False
            if not can_stop:
                return False, False
            return closing == "}", closing == "}"

        options = ENUM_SLOTS[index]
        remainder = text[pos:]
        terminator = LITERALS[index + 1]

        # Either still inside the value, or through it and into the next
        # literal. Try the longest legal value that the remainder starts with.
        matched = None
        for option in options:
            if remainder.startswith(option):
                if matched is None or len(option) > len(matched):
                    matched = option
        if matched is None:
            # Must at least be a prefix of some option.
            return any(option.startswith(remainder) for option in options), False

        # Ambiguity guard: a value can be a strict prefix of another (none in
        # the current schema, asserted by a test). If the remainder is exactly
        # the matched value we may also still be extending it.
        pos += len(matched)
        if pos == n:
            # Could stop here, or could be extending into a longer option.
            return True, False
        # fall through to the next literal on the next loop iteration
        continue

    return False, False


def is_valid_prefix(text: str) -> bool:
    return analyse(text)[0]


def is_complete(text: str) -> bool:
    return analyse(text)[1]


# --------------------------------------------------------------------------
# Logits processor
# --------------------------------------------------------------------------

def _allowed_characters() -> set[str]:
    chars: set[str] = set('{}",: ')
    for literal in LITERALS:
        chars.update(literal)
    for options in ENUM_SLOTS:
        for option in options:
            chars.update(option)
    chars.update("0123456789")
    return chars


def candidate_token_ids(tokenizer) -> list[int]:
    """Tokens that could ever appear in a legal output.

    Restricting the per-step scan to these (a few thousand of 151,936) is what
    makes exact masking affordable. It is a superset filter only: correctness
    still comes from the prefix automaton.
    """
    allowed = _allowed_characters()
    ids: list[int] = []
    vocab_size = len(tokenizer)
    for token_id in range(vocab_size):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if piece is None:
            continue
        text = tokenizer.convert_tokens_to_string([piece])
        if not text or any(ch not in allowed for ch in text):
            continue
        ids.append(token_id)
    return ids


class SchemaConstraint:
    """Per-sequence decoding constraint over the ForgeLM output language."""

    def __init__(self, tokenizer, candidate_ids: list[int] | None = None):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.candidates = (candidate_ids if candidate_ids is not None
                           else candidate_token_ids(tokenizer))
        self._token_text = {
            tid: tokenizer.convert_tokens_to_string(
                [tokenizer.convert_ids_to_tokens(tid)])
            for tid in self.candidates
        }
        # Memoised: prefix string -> allowed token ids. Sequences share long
        # prefixes (every output starts '{"category": "'), so this collapses
        # the work dramatically after the first few examples.
        self._cache: dict[str, list[int]] = {}
        self.stats = {"cache_hits": 0, "cache_misses": 0, "dead_ends": 0}

    def allowed_tokens(self, prefix: str) -> list[int]:
        cached = self._cache.get(prefix)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached
        self.stats["cache_misses"] += 1

        allowed = [tid for tid in self.candidates
                   if is_valid_prefix(prefix + self._token_text[tid])]
        self._cache[prefix] = allowed
        return allowed

    def mask_for(self, prefix: str) -> tuple[list[int], bool]:
        """Return (allowed token ids, may_finish_now)."""
        complete = is_complete(prefix)
        allowed = self.allowed_tokens(prefix)
        if complete:
            # Once the object is closed, the only legal continuation is to stop.
            return [self.eos_token_id], True
        if not allowed:
            # Unreachable if the automaton is correct; recorded rather than
            # silently producing a uniform distribution.
            self.stats["dead_ends"] += 1
            return [self.eos_token_id], True
        return allowed, False


def build_logits_processor(tokenizer, prompt_lengths: list[int],
                           constraint: SchemaConstraint | None = None):
    """A transformers LogitsProcessor enforcing the schema.

    `prompt_lengths` gives, per batch row, how many input tokens precede the
    generated portion, so the processor can decode only what the model produced.
    """
    import torch
    from transformers import LogitsProcessor

    shared = constraint or SchemaConstraint(tokenizer)

    class _SchemaLogitsProcessor(LogitsProcessor):
        def __init__(self) -> None:
            self.constraint = shared

        def __call__(self, input_ids, scores):
            for row in range(input_ids.shape[0]):
                generated = input_ids[row, prompt_lengths[row]:]
                prefix = self.constraint.tokenizer.decode(
                    generated, skip_special_tokens=True)
                allowed, _ = self.constraint.mask_for(prefix)

                mask = torch.full_like(scores[row], float("-inf"))
                index = torch.tensor(allowed, device=scores.device,
                                     dtype=torch.long)
                mask[index] = scores[row][index]
                scores[row] = mask
            return scores

    return _SchemaLogitsProcessor(), shared
