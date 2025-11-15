"""
Evaluation script to compute persona drift metrics from conversation logs.
"""

import argparse
import pickle
import json
from pathlib import Path
from statistics import mean
from typing import List, Dict, Optional, Any
import pandas as pd

from metrics import PersonaDriftMetrics
from utils import pkl2dict


def extract_agent_responses(pkl: Dict) -> List[str]:
    """Extract agent responses from conversation history."""
    history = pkl["history"]
    # Agent responses are at odd indices (1, 3, 5, ...)
    agent_responses = [history[i] for i in range(1, len(history), 2)]
    return agent_responses


def evaluate_conversation(pkl_path: Path, use_gpu: bool = False, early_turns: int = 3) -> Dict[str, Any]:
    """
    Evaluate a single conversation file.
    
    Returns dictionary with metrics.
    """
    with pkl_path.open("rb") as f:
        pkl = pickle.load(f)
    
    persona_desc = pkl["persona"]
    agent_responses = extract_agent_responses(pkl)
    
    if len(agent_responses) == 0:
        return {
            "error": "No agent responses found",
            "file": str(pkl_path)
        }
    
    # Initialize metrics calculator
    metrics_calc = PersonaDriftMetrics(persona_desc, use_gpu=use_gpu)
    
    # Compute all metrics
    all_metrics = metrics_calc.compute_all_metrics(agent_responses, early_turns=early_turns)
    persona_consistency = [float(x) for x in all_metrics["persona_consistency"]]
    contradiction_rate = [float(x) for x in all_metrics["contradiction_rate"]]
    drift_index = [float(x) for x in all_metrics["drift_index"]]
    conversation_quality = [float(x) for x in all_metrics["conversation_quality"]]
    
    # Aggregate statistics
    summary = pkl.get("summary", {})
    turn_stats = pkl.get("turn_stats", [])
    best_of_n_logs = pkl.get("best_of_n_logs", [])

    avg_latency = mean(stat["latency_sec"] for stat in turn_stats) if turn_stats else 0.0
    avg_prompt_tokens = mean(stat["prompt_tokens_est"] for stat in turn_stats) if turn_stats else 0.0
    avg_response_tokens = mean(stat["response_tokens_est"] for stat in turn_stats) if turn_stats else 0.0

    best_of_n_summary = None
    if best_of_n_logs:
        selected_stats = []
        all_candidates_counts = []
        for entry in best_of_n_logs:
            selected_idx = entry.get("selected_index")
            candidates = entry.get("candidates", [])
            if candidates:
                all_candidates_counts.append(len(candidates))
            if selected_idx is None or not candidates:
                continue
            for candidate in candidates:
                if candidate.get("index") == selected_idx:
                    selected_stats.append(candidate)
                    break
        if selected_stats:
            best_of_n_summary = {
                "mean_persona_similarity": float(mean(float(c["persona_similarity"]) for c in selected_stats)),
                "mean_context_similarity": float(mean(float(c["context_similarity"]) for c in selected_stats)),
                "mean_length_penalty": float(mean(float(c["length_penalty"]) for c in selected_stats)),
                "mean_generation_latency": float(mean(float(c.get("generation_latency", 0.0)) for c in selected_stats)),
                "mean_candidates_per_turn": float(mean(float(count) for count in all_candidates_counts))
                if all_candidates_counts
                else 0.0,
            }

    result = {
        "file": str(pkl_path),
        "model": pkl.get("model_name", "unknown"),
        "decoding_strategy": pkl.get("decoding_strategy", "unknown"),
        "num_turns": len(agent_responses),
        "persona": persona_desc[:100],  # Truncate for readability
        "summary": {
            "total_latency_sec": float(summary.get("total_latency_sec", 0.0)),
            "total_prompt_tokens_est": float(summary.get("total_prompt_tokens_est", 0.0)),
            "total_response_tokens_est": float(summary.get("total_response_tokens_est", 0.0)),
            "avg_latency_sec": float(avg_latency),
            "avg_prompt_tokens_est": float(avg_prompt_tokens),
            "avg_response_tokens_est": float(avg_response_tokens),
        },
        "best_of_n_summary": best_of_n_summary,
        "metrics": {
            "persona_consistency": {
                "mean": float(sum(persona_consistency) / len(persona_consistency)) if persona_consistency else 0.0,
                "std": float(pd.Series(persona_consistency).std()) if len(persona_consistency) > 1 else 0.0,
                "per_turn": persona_consistency,
            },
            "contradiction_rate": {
                "mean": float(sum(contradiction_rate) / len(contradiction_rate)) if contradiction_rate else 0.0,
                "std": float(pd.Series(contradiction_rate).std()) if len(contradiction_rate) > 1 else 0.0,
                "per_turn": contradiction_rate,
            },
            "drift_index": {
                "mean": float(sum(drift_index) / len(drift_index)) if drift_index else 0.0,
                "std": float(pd.Series(drift_index).std()) if len(drift_index) > 1 else 0.0,
                "per_turn": drift_index,
            },
            "conversation_quality": {
                "mean": float(sum(conversation_quality) / len(conversation_quality)) if conversation_quality else 0.0,
                "std": float(pd.Series(conversation_quality).std()) if len(conversation_quality) > 1 else 0.0,
                "per_turn": conversation_quality,
            },
        }
    }
    
    return result


def evaluate_directory(directory: Path, use_gpu: bool = False, early_turns: int = 3,
                       output_file: Optional[Path] = None) -> pd.DataFrame:
    """
    Evaluate all conversation files in a directory.
    
    Returns DataFrame with aggregated results.
    """
    pickle_files = list(directory.rglob("*.pkl"))
    
    if len(pickle_files) == 0:
        print(f"No pickle files found in {directory}")
        return pd.DataFrame()
    
    print(f"Found {len(pickle_files)} conversation file(s)")
    
    results = []
    for pkl_path in pickle_files:
        print(f"Evaluating: {pkl_path.name}")
        try:
            result = evaluate_conversation(pkl_path, use_gpu=use_gpu, early_turns=early_turns)
            if "error" not in result:
                results.append(result)
        except Exception as e:
            print(f"Error evaluating {pkl_path}: {e}")
            continue
    
    if len(results) == 0:
        print("No valid results to aggregate")
        return pd.DataFrame()
    
    # Convert to DataFrame
    rows = []
    for result in results:
        row = {
            "file": result["file"],
            "model": result["model"],
            "decoding_strategy": result["decoding_strategy"],
            "num_turns": result["num_turns"],
            "total_latency_sec": float(result["summary"]["total_latency_sec"]),
            "avg_latency_sec": float(result["summary"]["avg_latency_sec"]),
            "total_prompt_tokens_est": float(result["summary"]["total_prompt_tokens_est"]),
            "total_response_tokens_est": float(result["summary"]["total_response_tokens_est"]),
            "avg_prompt_tokens_est": float(result["summary"]["avg_prompt_tokens_est"]),
            "avg_response_tokens_est": float(result["summary"]["avg_response_tokens_est"]),
            "persona_consistency_mean": result["metrics"]["persona_consistency"]["mean"],
            "persona_consistency_std": result["metrics"]["persona_consistency"]["std"],
            "contradiction_rate_mean": result["metrics"]["contradiction_rate"]["mean"],
            "contradiction_rate_std": result["metrics"]["contradiction_rate"]["std"],
            "drift_index_mean": result["metrics"]["drift_index"]["mean"],
            "drift_index_std": result["metrics"]["drift_index"]["std"],
            "conversation_quality_mean": result["metrics"]["conversation_quality"]["mean"],
            "conversation_quality_std": result["metrics"]["conversation_quality"]["std"],
        }
        best_of_n_summary = result.get("best_of_n_summary")
        if best_of_n_summary:
            row.update(
                {
                    "best_of_n_mean_persona_similarity": best_of_n_summary["mean_persona_similarity"],
                    "best_of_n_mean_context_similarity": best_of_n_summary["mean_context_similarity"],
                    "best_of_n_mean_length_penalty": best_of_n_summary["mean_length_penalty"],
                    "best_of_n_mean_generation_latency": best_of_n_summary["mean_generation_latency"],
                    "best_of_n_mean_candidates_per_turn": best_of_n_summary["mean_candidates_per_turn"],
                }
            )
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save results
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Also save detailed JSON
        json_file = output_file.with_suffix('.json')
        with json_file.open('w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {json_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate persona drift metrics from conversation logs")
    parser.add_argument('--input', type=str, default='selfchat',
                       help='Input directory or file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (optional)')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU for metric computation')
    parser.add_argument('--early_turns', type=int, default=3,
                       help='Number of early turns to use as baseline for drift index')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Evaluate single file
        print(f"Evaluating single file: {input_path}")
        result = evaluate_conversation(input_path, use_gpu=args.use_gpu, early_turns=args.early_turns)
        print("\nResults:")
        print(json.dumps(result, indent=2))
    elif input_path.is_dir():
        # Evaluate directory
        output_path = Path(args.output) if args.output else None
        df = evaluate_directory(input_path, use_gpu=args.use_gpu, early_turns=args.early_turns,
                                output_file=output_path)
        
        if len(df) > 0:
            print("\nSummary Statistics:")
            print("=" * 80)
            print(df.describe())
            print("\nBy Decoding Strategy:")
            print("=" * 80)
            if "decoding_strategy" in df.columns:
                print(df.groupby("decoding_strategy")[
                    ["persona_consistency_mean", "contradiction_rate_mean", 
                     "drift_index_mean", "conversation_quality_mean"]
                ].mean())
    else:
        print(f"Error: {input_path} is not a valid file or directory")


if __name__ == '__main__':
    main()

