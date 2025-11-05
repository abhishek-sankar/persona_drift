"""Modal entrypoints for running persona_drift on modal.com.

Usage examples:

    modal run modal_app.py::run_selfchat --turns 8 --runs 1 --agent 0 --user 0

See the README for detailed setup instructions, including how to provide the
Replicate API token as a Modal secret and how to persist outputs to a Modal
volume.
"""

from __future__ import annotations

import modal


IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests", "langdetect", "datasets")
)

VOLUME = modal.Volume.from_name("persona-drift-selfchat", create_if_missing=True)

stub = modal.Stub("persona-drift", image=IMAGE)


def _build_args(
    *,
    model_name: str,
    agent: int,
    user: int,
    topic: int,
    seed: int,
    turns: int,
    runs: int,
) -> list[str]:
    return [
        "--model_name",
        model_name,
        "--agent",
        str(agent),
        "--user",
        str(user),
        "--topic",
        str(topic),
        "--seed",
        str(seed),
        "--turns",
        str(turns),
        "--runs",
        str(runs),
    ]


@stub.function(
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("replicate-api-token")],
    volumes={"/modal-selfchat": VOLUME},
)
def run_selfchat(
    *,
    model_name: str = "meta/llama-4-scout-instruct",
    agent: int = -1,
    user: int = -1,
    topic: int = -1,
    seed: int = 42,
    turns: int = 16,
    runs: int = 1,
) -> None:
    """Execute the self-chat baseline within Modal.

    Args mirror ``run.py``. Outputs are persisted to the Modal volume named
    ``persona-drift-selfchat`` under ``/modal-selfchat``.
    """

    import os
    from pathlib import Path

    import run as selfchat

    output_dir = Path("/modal-selfchat")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SELFCHAT_DIR", str(output_dir))

    args = _build_args(
        model_name=model_name,
        agent=agent,
        user=user,
        topic=topic,
        seed=seed,
        turns=turns,
        runs=runs,
    )

    selfchat.main(args)

