"""Modal entrypoints for running persona_drift on modal.com.

Usage examples:

    modal run modal_app.py::run_selfchat --turns 8 --runs 1 --agent 0 --user 0

See the README for detailed setup instructions, including how to provide the
Replicate API token as a Modal secret and how to persist outputs to a Modal
volume.
"""

from __future__ import annotations

import modal

# Build the image with all necessary dependencies and local files
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "requests",
        "langdetect",
        "datasets",
        "nltk",  # Optional but preferred for full dataset
    )
    .add_local_dir(".", remote_path="/workspace")  # Mount project files
)

VOLUME = modal.Volume.from_name("persona-drift-selfchat", create_if_missing=True)

app = modal.App("persona-drift", image=IMAGE)


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


@app.function(
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("replicate-api-token")],
    volumes={"/modal-selfchat": VOLUME},
    image=IMAGE,
)
def run_selfchat(
    model_name: str = "meta/llama-2-70b-chat",
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
    import sys
    from pathlib import Path

    # Set output directory BEFORE importing run.py
    # run.py sets SELFCHAT_DIR at module import time, so we must set it first
    output_dir = Path("/modal-selfchat")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SELFCHAT_DIR"] = str(output_dir)

    # Add the workspace directory to Python path
    sys.path.insert(0, "/workspace")

    # Change working directory to workspace to ensure relative imports work
    os.chdir("/workspace")
    
    # Download NLTK data if not already available
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words', quiet=True)

    import run as selfchat

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
    
    # Commit volume changes to persist them
    VOLUME.commit()


@app.function(
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("replicate-api-token")],
    volumes={"/modal-selfchat": VOLUME},
    image=IMAGE,
)
def list_outputs() -> dict:
    """List all output files in the volume."""
    from pathlib import Path
    
    VOLUME.reload()  # Reload to see latest changes
    output_dir = Path("/modal-selfchat")
    
    files = {}
    if output_dir.exists():
        for file_path in output_dir.rglob("*.pkl"):
            rel_path = file_path.relative_to(output_dir)
            files[str(rel_path)] = file_path.stat().st_size
    
    return files


@app.function(
    volumes={"/modal-selfchat": VOLUME},
    image=IMAGE,
)
def download_file(remote_path: str) -> bytes:
    """Download a file from the volume."""
    from pathlib import Path
    
    VOLUME.reload()
    file_path = Path("/modal-selfchat") / remote_path
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {remote_path}")
    
    return file_path.read_bytes()

