#!/usr/bin/env python3
"""Helper script to view and analyze persona drift results."""

import pickle
import sys
from pathlib import Path
from utils import pkl2script, pkl2dict

def view_pickle_file(file_path: Path):
    """View the contents of a pickle file."""
    print(f"\n{'='*80}")
    print(f"File: {file_path}")
    print(f"{'='*80}\n")
    
    with file_path.open("rb") as f:
        pkl = pickle.load(f)
    
    print(f"Topic: {pkl['topic']}")
    print(f"Persona: {pkl['persona']}")
    print(f"User: {pkl['user']}")
    print(f"Seed: {pkl['seed']}")
    print(f"\nConversation History ({len(pkl['history'])} turns):")
    print("-" * 80)
    
    script = pkl2script(pkl)
    print(script)
    
    print(f"\n{'='*80}")
    print("Probed History Per Turn:")
    print(f"{'='*80}")
    for turn, answers in pkl.get("probed_history_per_turn", {}).items():
        print(f"\nTurn {turn} ({len(answers)} probe(s)):")
        for i, answer in enumerate(answers, 1):
            print(f"  Probe {i}: {answer}")

def list_all_pickles(directory: Path):
    """List all pickle files in a directory."""
    print(f"\nSearching for pickle files in: {directory}")
    print("-" * 80)
    
    pickle_files = list(directory.rglob("*.pkl"))
    
    if not pickle_files:
        print("No pickle files found!")
        return []
    
    print(f"Found {len(pickle_files)} pickle file(s):\n")
    for i, pkl_file in enumerate(pickle_files, 1):
        rel_path = pkl_file.relative_to(directory)
        size = pkl_file.stat().st_size / 1024  # Size in KB
        print(f"{i}. {rel_path} ({size:.2f} KB)")
    
    return pickle_files

def main():
    """Main function."""
    if len(sys.argv) > 1:
        # View specific file
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        view_pickle_file(file_path)
    else:
        # List all files and prompt to view one
        search_dirs = [
            Path("local-selfchat"),
            Path("selfchat"),
            Path("."),
        ]
        
        pickle_files = []
        for directory in search_dirs:
            if directory.exists():
                pickle_files.extend(directory.rglob("*.pkl"))
        
        if not pickle_files:
            print("No pickle files found in:")
            for d in search_dirs:
                print(f"  - {d}")
            print("\nTry downloading files from Modal volume:")
            print("  modal volume get persona-drift-selfchat /meta ./local-selfchat")
            sys.exit(1)
        
        print(f"\nFound {len(pickle_files)} pickle file(s):\n")
        for i, pkl_file in enumerate(pickle_files, 1):
            print(f"{i}. {pkl_file}")
        
        if len(pickle_files) == 1:
            print(f"\nViewing the only file found:")
            view_pickle_file(pickle_files[0])
        else:
            print(f"\nTo view a specific file, run:")
            print(f"  python view_results.py <path_to_file.pkl>")
            print(f"\nExample:")
            print(f"  python view_results.py {pickle_files[0]}")

if __name__ == "__main__":
    main()

