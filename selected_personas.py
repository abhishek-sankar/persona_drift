"""
Selected subset of persona prompts for baseline experiments.

We import the full persona catalog from `hundred_system_prompts` when available
so that judge functions and supporting utilities (e.g., sentiment scoring) are
preserved. If the heavy dependencies required by that module are unavailable,
we gracefully fall back to the smaller personas defined in `minimal_prompts`.
"""

from __future__ import annotations

from typing import Iterable, Tuple

try:  # Prefer the full benchmark personas
    from hundred_system_prompts import persona_system_prompts
except ImportError:  # pragma: no cover - fallback for lightweight setups
    from minimal_prompts import persona_system_prompts  # type: ignore

# Choose 20 diverse personas spanning affect, style, and behavioural traits.
# Indices refer to entries inside `persona_system_prompts`.
SELECTED_PERSONA_INDICES: Tuple[int, ...] = (
    0,   # Tennis enthusiast
    1,   # Rare vocabulary
    2,   # Rude persona
    3,   # Very happy
    4,   # Very sad
    5,   # DAN / malicious AI
    6,   # Teen slang
    7,   # Unhelpful assistant
    8,   # Incorrect math answers
    9,   # Cat persona
    10,  # Extremely long responses
    11,  # Architecture enthusiast
    12,  # Loving parent
    13,  # Thankful responder
    14,  # Book recommendation memory
    15,  # Father of three
    16,  # Father of 97
    17,  # Mention Tokyo
    18,  # Mention printers
    19,  # Secret number sharing
)

available_count = len(persona_system_prompts)
valid_indices = tuple(idx for idx in SELECTED_PERSONA_INDICES if 0 <= idx < available_count)

if len(valid_indices) < len(SELECTED_PERSONA_INDICES):
    print(
        "[selected_personas] Warning: full persona catalog unavailable; "
        f"using {len(valid_indices)} personas from fallback module."
    )

selected_personas = tuple(persona_system_prompts[idx] for idx in valid_indices)

if len(selected_personas) == 0:
    raise RuntimeError("No personas available; check installation of hundred_system_prompts dependencies.")


def get_persona_by_id(persona_id: int):
    """Return the persona triple (system prompt, probe, judge) for the given ID."""
    if persona_id < 0 or persona_id >= len(selected_personas):
        raise ValueError(f"Persona ID must be between 0 and {len(selected_personas) - 1}")
    return selected_personas[persona_id]


def iterate_personas() -> Iterable[tuple[str, str, object]]:
    """Yield all selected personas."""
    return iter(selected_personas)


NUM_PERSONAS = len(selected_personas)


