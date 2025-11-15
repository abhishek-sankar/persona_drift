import os

import argparse
import copy
import importlib
import importlib.util
import json
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import List, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from utils import *

SELFCHAT_DIR = Path(os.environ.get("SELFCHAT_DIR", "selfchat"))
SELFCHAT_DIR.mkdir(parents=True, exist_ok=True)

_full_dataset_available = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("nltk", "langdetect", "requests")
)

if _full_dataset_available:
    try:
        from hundred_system_prompts import (
            pattern_system_prompts,
            multiple_choice_system_prompts,
            persona_system_prompts,
            memorization_system_prompts,
            language_system_prompts,
        )
    except (ImportError, Exception) as e:
        # Fall back to minimal_prompts if hundred_system_prompts fails to import
        # (e.g., due to network errors downloading external resources)
        print(f"Warning: Failed to import hundred_system_prompts ({e}), falling back to minimal_prompts")
        from minimal_prompts import (
            pattern_system_prompts,
            multiple_choice_system_prompts,
            persona_system_prompts,
            memorization_system_prompts,
            language_system_prompts,
        )
else:
    from minimal_prompts import (
        pattern_system_prompts,
        multiple_choice_system_prompts,
        persona_system_prompts,
        memorization_system_prompts,
        language_system_prompts,
    )

index_list = [0, 0, 0, 0, 0]
personas = [_[__] for _, __ in zip([pattern_system_prompts, multiple_choice_system_prompts, persona_system_prompts, memorization_system_prompts, language_system_prompts], index_list)]
other_personas = [_[__:] for _, __ in zip([pattern_system_prompts, multiple_choice_system_prompts, persona_system_prompts, memorization_system_prompts, language_system_prompts], [1, 1, 1, 1, 1])]
for _ in other_personas:
    personas.extend(_)

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    parser.add_argument('--agent', type=int, default=-1, choices=[-1, ] + list(range(len(personas))))
    parser.add_argument('--user', type=int, default=-1, choices=[-1, ] + list(range(len(personas))))
    parser.add_argument('--topic', type=int, default=-1, choices=[-1] + list(range(len(topics))))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--turns', type=int, default=16)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args(argv)

    random.seed(args.seed)

    def seed_optional(module_name: str, attr_path: str, seed_value: int) -> None:
        if importlib.util.find_spec(module_name) is None:
            return
        module = importlib.import_module(module_name)
        target = module
        for attr in attr_path.split('.'):
            target = getattr(target, attr)
        target(seed_value)

    seed_optional("torch", "manual_seed", args.seed)
    seed_optional("numpy", "random.seed", args.seed)
    
    if args.agent == -1:
        args.agent = random.randint(0, len(personas)-1)
    if args.user == -1:
        args.user = random.randint(0, len(personas)-1)
    persona, probe_str, judge_func = personas[args.agent]
    user, probe_str_user, judge_func_user = personas[args.user]
    if args.topic == -1:
        args.topic = random.randint(0, len(topics)-1)
    topic = topics[args.topic]
    print(f"Now {args.model_name} chatting over {topic} with system prompts: (A) {persona} and (B) {user}")

    # load assistant
    use_api = "gpt" in args.model_name
    if use_api:
        from openai import OpenAI

        client = OpenAI()
    else:
        replicate_model = ENGINE_MAP.get(args.model_name, args.model_name)
        replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not replicate_api_token:
            raise EnvironmentError(
                "REPLICATE_API_TOKEN environment variable must be set to call Replicate models."
            )
        headers = {
            "Authorization": f"Token {replicate_api_token}",
            "Content-Type": "application/json",
        }
        create_url = f"https://api.replicate.com/v1/models/{replicate_model}/predictions"

        def replicate_request(method: str, url: str, payload: dict | None = None) -> dict:
            data = None
            if payload is not None:
                data = json.dumps(payload).encode("utf-8")
            req = urllib_request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib_request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except HTTPError as err:
                error_body = err.read().decode("utf-8", errors="ignore")
                raise RuntimeError(
                    f"Replicate API request failed with status {err.code}: {error_body}"
                ) from err
            except URLError as err:
                raise RuntimeError(f"Replicate API request failed: {err.reason}") from err

        def generate_with_replicate(prompt_text: str) -> str:
            prediction = replicate_request(
                "POST",
                create_url,
                {
                    "input": {
                        "prompt": prompt_text,
                        "max_tokens": 400,
                        "temperature": 1.0,
                        "top_p": 0.9,
                        "presence_penalty": 0,
                        "frequency_penalty": 0,
                    }
                },
            )
            prediction_id = prediction["id"]
            status = prediction.get("status", "")

            while status not in {"succeeded", "failed", "canceled"}:
                time.sleep(1.5)
                prediction = replicate_request(
                    "GET",
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                )
                status = prediction.get("status", "")

            if status != "succeeded":
                error_message = prediction.get("error", "unknown error")
                raise RuntimeError(
                    f"Replicate prediction {prediction_id} finished with status '{status}': {error_message}"
                )

            output = prediction.get("output", "")
            if isinstance(output, list):
                return "".join(str(chunk) for chunk in output)
            if output is None:
                return ""
            return str(output)
        
    # task management
    file_name = (
        f"{args.model_name}_agent_{args.agent}_user_{args.user}_turn_{args.turns}.pkl"
    )
    output_path = SELFCHAT_DIR / file_name
    # Ensure parent directories exist (model_name may contain slashes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:  # resuming halfway jobs if possible
        with output_path.open("rb") as handle:
            old_pkl = pickle.load(handle)
        pkl = {
            "topic": topic, 
            "history": old_pkl["history"], 
            "probed_history_per_turn": old_pkl["probed_history_per_turn"],
            "seed": args.seed, 
            "persona": persona, 
            "user": user,
        }
    except:
        pkl = {
            "topic": topic, 
            "history": [topic], 
            "probed_history_per_turn": defaultdict(list),
            "seed": args.seed, 
            "persona": persona, 
            "user": user,
        }
    
    for turn in range(len(pkl["history"])+1, args.turns+1):
        pkl_copy = copy.deepcopy(pkl)
        tick = time.time()
        messages = pkl2dict(pkl_copy)
        prompt = llama_v2_prompt(messages)
        print("@"*100)
        print(f"Prompting for the {turn}-th (one-based) turn with prompt:\n{prompt}")
        if use_api:
            completion = client.chat.completions.create(model=args.model_name, messages=messages)
            sequence = completion.choices[0].message.content
        else:
            sequence = generate_with_replicate(prompt)
        pkl["history"].append(process_answer(sequence))
        tok = time.time()
        print(f"Time taken for turn {turn}: {tok-tick:.2f} seconds")
        if len(pkl["history"]) % 2 == 0:
            with output_path.open("wb") as handle:
                pickle.dump(pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for turn in range(2, args.turns+1, 2):  # for 2, 4, 6, 8, 10, ...
        runs_to_run = args.runs - len(pkl["probed_history_per_turn"][turn])
        for _ in range(runs_to_run):
            temp_pkl = copy.deepcopy(pkl)
            temp_pkl["history"] = temp_pkl["history"][:turn]
            temp_pkl["history"].append(probe_str)
            pkl_copy = copy.deepcopy(temp_pkl)
            tick = time.time()
            messages = pkl2dict(pkl_copy)
            prompt = llama_v2_prompt(messages)
            if use_api:
                completion = client.chat.completions.create(model=args.model_name, messages=messages)
                sequence = completion.choices[0].message.content
            else:
                sequence = generate_with_replicate(prompt)
            pkl["probed_history_per_turn"][turn].append(process_answer(sequence))
            tok = time.time()
            print(f"Time taken for probe turn {turn} ({_+1}/{runs_to_run}): {tok-tick:.2f} seconds")

        with output_path.open("wb") as handle:
            pickle.dump(pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    pprint(f"Saved to {output_path}")

if __name__ == '__main__':
    main()
