"""
Baseline self-chat driver that uses Replicate-hosted models exclusively.
Supports greedy, nucleus, and best-of-n decoding with persona-aware re-ranking.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

import numpy as np  # type: ignore[import]

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
except ImportError:
    SentenceTransformer = None

try:
    from transformers import AutoTokenizer  # type: ignore[import]
except ImportError:
    AutoTokenizer = None

from utils import (
    ENGINE_MAP,
    llama_v2_prompt,
    pkl2dict,
    process_answer,
    topics,
)
from selected_personas import get_persona_by_id, NUM_PERSONAS


SELFCHAT_DIR = Path(os.environ.get("SELFCHAT_DIR", "selfchat"))
SELFCHAT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DecodingStrategy:
    name: str = field(init=False)

    def get_generation_params(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class GreedyDecoding(DecodingStrategy):
    def __post_init__(self):
        self.name = "greedy"

    def get_generation_params(self) -> Dict[str, Any]:
        return {"temperature": 0.0, "top_p": 1.0}


@dataclass
class NucleusSampling(DecodingStrategy):
    top_p: float = 0.9
    temperature: float = 0.7

    def __post_init__(self):
        self.name = f"nucleus_p{self.top_p}_t{self.temperature}"

    def get_generation_params(self) -> Dict[str, Any]:
        return {"temperature": self.temperature, "top_p": self.top_p}


@dataclass
class BestOfNDecoding(NucleusSampling):
    n: int = 3
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.0

    def __post_init__(self):
        if self.n < 1:
            raise ValueError("best-of-n requires n>=1")
        self.name = f"nucleus_p{self.top_p}_t{self.temperature}"
        self.name = f"bestof{self.n}_p{self.top_p}_t{self.temperature}"


class SentenceEmbeddingHelper:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for best-of-n re-ranking. "
                "Install it with `pip install sentence-transformers`."
            )
        self.model = SentenceTransformer(self.MODEL_NAME)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)


class TokenCounter:
    def __init__(self, tokenizer_name: Optional[str] = None):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        if tokenizer_name and AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                print(f"[TokenCounter] Using tokenizer '{tokenizer_name}' for token counting.")
            except Exception as exc:  # pragma: no cover - warning path
                print(f"[TokenCounter] Failed to load tokenizer '{tokenizer_name}': {exc}. Falling back to word counts.")
                self.tokenizer = None
        elif tokenizer_name:
            print("[TokenCounter] transformers not installed; falling back to word counts.")

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return len(text.split())


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8
    return float(np.dot(vec_a, vec_b) / denom)


def compute_length_penalty(text: str) -> float:
    return float(len(text.split()))


def replicate_generate(
    prompt_text: str,
    *,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    poll_interval: float,
) -> str:
    replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not replicate_api_token:
        raise EnvironmentError("REPLICATE_API_TOKEN environment variable must be set.")

    replicate_model = ENGINE_MAP.get(model_name, model_name)
    create_url = f"https://api.replicate.com/v1/models/{replicate_model}/predictions"
    headers = {
        "Authorization": f"Token {replicate_api_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "input": {
            "prompt": prompt_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }
    }

    req = urllib_request.Request(create_url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=120) as resp:
            prediction = json.loads(resp.read().decode("utf-8"))
    except HTTPError as err:
        error_body = err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Replicate API request failed: {error_body}") from err
    except URLError as err:
        raise RuntimeError(f"Replicate API request failed: {err.reason}") from err

    prediction_id = prediction["id"]
    status = prediction.get("status", "")

    while status not in {"succeeded", "failed", "canceled"}:
        time.sleep(poll_interval)
        status_req = urllib_request.Request(
            f"https://api.replicate.com/v1/predictions/{prediction_id}",
            headers=headers,
            method="GET",
        )
        with urllib_request.urlopen(status_req, timeout=60) as resp:
            prediction = json.loads(resp.read().decode("utf-8"))
        status = prediction.get("status", "")

    if status != "succeeded":
        error_message = prediction.get("error", "unknown error")
        raise RuntimeError(f"Replicate prediction failed: {error_message}")

    output = prediction.get("output", "")
    if isinstance(output, list):
        return "".join(str(chunk) for chunk in output)
    if output is None:
        return ""
    return str(output)


def best_of_n_generate(
    prompt_text: str,
    strategy: BestOfNDecoding,
    *,
    model_name: str,
    max_tokens: int,
    poll_interval: float,
    persona_desc: str,
    history: List[str],
    embedding_helper: SentenceEmbeddingHelper,
    persona_embedding: np.ndarray,
) -> Tuple[str, Dict[str, Any]]:
    params = strategy.get_generation_params()

    candidates: List[str] = []
    candidate_latencies: List[float] = []
    for _ in range(strategy.n):
        start = time.time()
        text = replicate_generate(
            prompt_text,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=params["temperature"],
            top_p=params["top_p"],
            poll_interval=poll_interval,
        )
        candidate_latencies.append(time.time() - start)
        candidates.append(text)

    context_text = " ".join(history) or persona_desc
    context_embedding = embedding_helper.encode([context_text])[0]
    candidate_embeddings = embedding_helper.encode(candidates)

    scored_candidates: List[Dict[str, Any]] = []
    for idx, cand_text in enumerate(candidates):
        cand_emb = candidate_embeddings[idx]
        persona_sim = float(cosine_similarity(cand_emb, persona_embedding))
        context_sim = float(cosine_similarity(cand_emb, context_embedding))
        length_penalty = float(compute_length_penalty(cand_text))
        total_score = float(
            strategy.alpha * persona_sim
            + strategy.beta * context_sim
            + strategy.gamma * length_penalty
        )
        scored_candidates.append(
            {
                "index": idx,
                "score": total_score,
                "persona_similarity": persona_sim,
                "context_similarity": context_sim,
                "length_penalty": length_penalty,
                "candidate_length": len(cand_text.split()),
                "generation_latency": float(candidate_latencies[idx]),
            }
        )

    best_entry = max(scored_candidates, key=lambda item: item["score"])
    best_idx = best_entry["index"]
    selected_text = candidates[best_idx]

    metadata = {
        "selected_index": best_idx,
        "candidates": scored_candidates,
    }
    return selected_text, metadata


def prepare_output_file(model_name: str, agent: int, user: int, turns: int, strategy_name: str, seed: int) -> Path:
    file_name = (
        f"{model_name.replace('/', '_')}_agent_{agent}_user_{user}_"
        f"turn_{turns}_{strategy_name}_seed_{seed}.pkl"
    )
    output_path = SELFCHAT_DIR / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Persona drift self-chat via Replicate models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta/llama-4-scout-instruct",
        help="Replicate model name (or alias defined in utils.ENGINE_MAP).",
    )
    parser.add_argument(
        "--agent",
        type=int,
        default=-1,
        help=f"Agent persona ID (0-{NUM_PERSONAS-1}; -1 samples randomly).",
    )
    parser.add_argument(
        "--user",
        type=int,
        default=-1,
        help=f"User persona ID (0-{NUM_PERSONAS-1}; -1 samples randomly).",
    )
    parser.add_argument("--topic", type=int, default=-1, help="Topic ID (-1 selects randomly).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turns", type=int, default=20, help="Total turns (including both speakers).")
    parser.add_argument(
        "--decoding",
        type=str,
        default="greedy",
        choices=["greedy", "nucleus", "best_of_n"],
        help="Decoding strategy.",
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p value for nucleus/best-of-n.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for nucleus/best-of-n.")
    parser.add_argument("--best_of_n", type=int, default=3, help="Number of samples for best-of-n.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Persona similarity weight for best-of-n.")
    parser.add_argument("--beta", type=float, default=0.5, help="Context similarity weight for best-of-n.")
    parser.add_argument("--gamma", type=float, default=0.0, help="Length penalty weight for best-of-n.")
    parser.add_argument("--max_tokens", type=int, default=400, help="Max tokens per response (Replicate setting).")
    parser.add_argument("--poll_interval", type=float, default=1.5, help="Polling interval (seconds) for Replicate.")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Optional Hugging Face tokenizer for token counting.",
    )
    parser.add_argument("--log_every", type=int, default=2, help="Persist conversation after this many turns.")

    args = parser.parse_args(argv)

    random.seed(args.seed)

    if args.agent < 0 or args.agent >= NUM_PERSONAS:
        args.agent = random.randint(0, NUM_PERSONAS - 1)
    if args.user < 0 or args.user >= NUM_PERSONAS:
        args.user = random.randint(0, NUM_PERSONAS - 1)

    persona_desc, _, _ = get_persona_by_id(args.agent)
    user_desc, _, _ = get_persona_by_id(args.user)

    if args.topic == -1:
        args.topic = random.randint(0, len(topics) - 1)
    topic = topics[args.topic]

    if args.decoding == "greedy":
        strategy: DecodingStrategy = GreedyDecoding()
    elif args.decoding == "nucleus":
        strategy = NucleusSampling(top_p=args.top_p, temperature=args.temperature)
    else:
        strategy = BestOfNDecoding(
            n=args.best_of_n,
            top_p=args.top_p,
            temperature=args.temperature,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )
        embedding_helper = SentenceEmbeddingHelper()
        persona_embedding = embedding_helper.encode([persona_desc])[0]

    token_counter = TokenCounter(args.tokenizer_name)
    output_path = prepare_output_file(
        args.model_name,
        args.agent,
        args.user,
        args.turns,
        strategy.name,
        args.seed,
    )

    try:
        with output_path.open("rb") as handle:
            pkl = pickle.load(handle)
        print(f"[Resume] Loaded existing conversation with {len(pkl.get('history', [])) - 1} completed turns.")
    except FileNotFoundError:
        pkl = {
            "topic": topic,
            "history": [topic],
            "seed": args.seed,
            "persona_id": args.agent,
            "user_id": args.user,
            "persona": persona_desc,
            "user": user_desc,
            "decoding_strategy": strategy.name,
            "model_name": args.model_name,
            "config": vars(args),
            "turn_stats": [],
            "best_of_n_logs": [],
        }
    except Exception as exc:
        print(f"[Warning] Failed to load previous state ({exc}); starting a fresh run.")
        pkl = {
            "topic": topic,
            "history": [topic],
            "seed": args.seed,
            "persona_id": args.agent,
            "user_id": args.user,
            "persona": persona_desc,
            "user": user_desc,
            "decoding_strategy": strategy.name,
            "model_name": args.model_name,
            "config": vars(args),
            "turn_stats": [],
            "best_of_n_logs": [],
        }

    pkl.setdefault("turn_stats", [])
    pkl.setdefault("best_of_n_logs", [])

    print(f"Model: {args.model_name} (Replicate)")
    print(f"Agent persona [{args.agent}]: {persona_desc}")
    print(f"User persona  [{args.user}]: {user_desc}")
    print(f"Topic: {topic}")
    print(f"Decoding strategy: {strategy.name}")
    print(f"Turns: {args.turns}")

    for turn in range(len(pkl["history"]), args.turns + 1):
        pkl_copy = copy.deepcopy(pkl)
        messages = pkl2dict(pkl_copy)
        prompt_text = llama_v2_prompt(messages)

        print(f"\n{'=' * 80}")
        print(f"Turn {turn}/{args.turns}")
        print(f"{'=' * 80}")

        turn_start = time.time()

        if isinstance(strategy, BestOfNDecoding):
            sequence, metadata = best_of_n_generate(
                prompt_text,
                strategy,
                model_name=args.model_name,
                max_tokens=args.max_tokens,
                poll_interval=args.poll_interval,
                persona_desc=persona_desc,
                history=pkl["history"],
                embedding_helper=embedding_helper,
                persona_embedding=persona_embedding,
            )
            metadata["turn"] = turn
            pkl["best_of_n_logs"].append(metadata)
        else:
            params = strategy.get_generation_params()
            sequence = replicate_generate(
                prompt_text,
                model_name=args.model_name,
                max_tokens=args.max_tokens,
                temperature=params["temperature"],
                top_p=params["top_p"],
                poll_interval=args.poll_interval,
            )

        latency = time.time() - turn_start
        response_text = process_answer(sequence)
        pkl["history"].append(response_text)

        prompt_tokens = token_counter.count(prompt_text)
        response_tokens = token_counter.count(response_text)
        pkl["turn_stats"].append(
            {
                "turn": turn,
                "prompt_tokens_est": prompt_tokens,
                "response_tokens_est": response_tokens,
                "latency_sec": latency,
            }
        )

        print(f"Response: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
        print(f"Latency: {latency:.2f}s | prompt tokens≈{prompt_tokens} | response tokens≈{response_tokens}")

        if turn % max(args.log_every, 1) == 0:
            with output_path.open("wb") as handle:
                pickle.dump(pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)

    total_latency = sum(stat["latency_sec"] for stat in pkl["turn_stats"])
    total_prompt_tokens = sum(stat["prompt_tokens_est"] for stat in pkl["turn_stats"])
    total_response_tokens = sum(stat["response_tokens_est"] for stat in pkl["turn_stats"])
    pkl["summary"] = {
        "total_turns": len(pkl["history"]) - 1,
        "total_latency_sec": total_latency,
        "total_prompt_tokens_est": total_prompt_tokens,
        "total_response_tokens_est": total_response_tokens,
    }

    with output_path.open("wb") as handle:
        pickle.dump(pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n{'=' * 80}")
    print(f"Conversation complete! Saved to: {output_path}")
    print(
        f"Total latency: {total_latency:.2f}s | prompt tokens≈{total_prompt_tokens} | "
        f"response tokens≈{total_response_tokens}"
    )
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

