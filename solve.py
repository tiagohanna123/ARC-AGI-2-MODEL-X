#!/usr/bin/env python3
"""
ARC-AGI-2 Model X Solver - Main Entry Point

equilibriumai v2 - Created by Tiago Hanna
https://github.com/tiagohanna123/o

Usage:
    python solve.py                     # Solve all evaluation tasks
    python solve.py path/to/task.json   # Solve a specific task
    python solve.py --sample            # Run on sample tasks
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any

from model_x.solver import ARCSolver, solve_arc_task, evaluate_task
from model_x.system_prompt import get_system_prompt, get_arc_agi_prompt


def get_data_dir() -> str:
    """Get the data directory path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'data')


def list_tasks(subset: str = 'evaluation') -> List[str]:
    """List all task files in a subset."""
    data_dir = get_data_dir()
    subset_dir = os.path.join(data_dir, subset)
    
    if not os.path.exists(subset_dir):
        print(f"warning: directory {subset_dir} not found")
        return []
    
    tasks = [f for f in os.listdir(subset_dir) if f.endswith('.json')]
    return sorted(tasks)


def solve_single_task(task_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Solve a single task and return results."""
    if verbose:
        print(f"processing: {os.path.basename(task_path)}")
    
    try:
        result = evaluate_task(task_path)
        if verbose:
            correct_count = sum(result.get('correct', []))
            total = len(result.get('correct', []))
            status = 'solved' if correct_count == total and total > 0 else 'attempted'
            print(f"  status: {status}")
            if result.get('correct'):
                print(f"  correct: {correct_count}/{total}")
        return result
    except Exception as e:
        if verbose:
            print(f"  error: {str(e)}")
        return {
            'task_file': os.path.basename(task_path),
            'error': str(e)
        }


def solve_all_tasks(subset: str = 'evaluation', 
                    verbose: bool = True) -> List[Dict[str, Any]]:
    """Solve all tasks in a subset."""
    data_dir = get_data_dir()
    subset_dir = os.path.join(data_dir, subset)
    
    tasks = list_tasks(subset)
    results = []
    
    print(f"\nequilibriumai v2 - arc-agi-2 solver")
    print(f"processing {len(tasks)} {subset} tasks\n")
    
    for task_name in tasks:
        task_path = os.path.join(subset_dir, task_name)
        result = solve_single_task(task_path, verbose)
        results.append(result)
    
    return results


def solve_sample_tasks(verbose: bool = True) -> List[Dict[str, Any]]:
    """Solve a sample of tasks to demonstrate functionality."""
    data_dir = get_data_dir()
    eval_dir = os.path.join(data_dir, 'evaluation')
    
    # Select first 5 evaluation tasks as samples
    tasks = list_tasks('evaluation')[:5]
    results = []
    
    print("\nequilibriumai v2 - arc-agi-2 solver (sample mode)")
    print(f"processing {len(tasks)} sample tasks\n")
    
    for task_name in tasks:
        task_path = os.path.join(eval_dir, task_name)
        result = solve_single_task(task_path, verbose)
        results.append(result)
    
    return results


def generate_submission(output_path: str = 'submission.json'):
    """Generate a submission file for all evaluation tasks."""
    data_dir = get_data_dir()
    eval_dir = os.path.join(data_dir, 'evaluation')
    
    tasks = list_tasks('evaluation')
    submission = {}
    
    solver = ARCSolver()
    
    print("\nequilibriumai v2 - generating submission")
    print(f"processing {len(tasks)} evaluation tasks\n")
    
    for task_name in tasks:
        task_path = os.path.join(eval_dir, task_name)
        task_id = task_name.replace('.json', '')
        
        try:
            predictions = solver.solve_file(task_path)
            # Format: each test input gets 2 attempts
            submission[task_id] = []
            for pred in predictions:
                submission[task_id].append({
                    'attempt_1': pred,
                    'attempt_2': pred  # Same prediction for both attempts
                })
            print(f"  {task_id}: {len(predictions)} prediction(s)")
        except Exception as e:
            print(f"  {task_id}: error - {str(e)}")
            submission[task_id] = []
    
    # Write submission file
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"\nsubmission saved to: {output_path}")
    return submission


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary of results."""
    total = len(results)
    errors = sum(1 for r in results if 'error' in r)
    with_answers = sum(1 for r in results if r.get('correct'))
    correct = sum(
        sum(r.get('correct', [])) 
        for r in results 
        if r.get('correct')
    )
    total_tests = sum(
        len(r.get('correct', [])) 
        for r in results 
        if r.get('correct')
    )
    
    print("\n" + "=" * 50)
    print("summary")
    print("=" * 50)
    print(f"total tasks: {total}")
    print(f"tasks with errors: {errors}")
    if total_tests > 0:
        print(f"test cases evaluated: {total_tests}")
        print(f"correct predictions: {correct}")
        print(f"accuracy: {100 * correct / total_tests:.1f}%")
    print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='ARC-AGI-2 Model X Solver - equilibriumai v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python solve.py                      solve all evaluation tasks
  python solve.py task.json            solve a specific task
  python solve.py --sample             run on sample tasks
  python solve.py --submit             generate submission file
  python solve.py --prompt             show system prompt
        """
    )
    
    parser.add_argument('task', nargs='?', help='path to a specific task json file')
    parser.add_argument('--sample', action='store_true', 
                       help='run on sample tasks only')
    parser.add_argument('--submit', action='store_true', 
                       help='generate submission.json file')
    parser.add_argument('--prompt', action='store_true', 
                       help='display the system prompt')
    parser.add_argument('--training', action='store_true', 
                       help='run on training tasks instead of evaluation')
    parser.add_argument('-q', '--quiet', action='store_true', 
                       help='minimal output')
    
    args = parser.parse_args()
    
    if args.prompt:
        print("equilibriumai v2 - system prompt\n")
        print("-" * 50)
        print(get_system_prompt())
        print("-" * 50)
        print("\narc-agi mode prompt\n")
        print("-" * 50)
        print(get_arc_agi_prompt())
        print("-" * 50)
        return
    
    if args.task:
        # Solve a specific task
        result = solve_single_task(args.task, verbose=not args.quiet)
        if not args.quiet:
            print(f"\npredictions:")
            for i, pred in enumerate(result.get('predictions', [])):
                print(f"\ntest case {i + 1}:")
                for row in pred:
                    print('  ' + ' '.join(str(c) for c in row))
        return
    
    if args.submit:
        generate_submission()
        return
    
    if args.sample:
        results = solve_sample_tasks(verbose=not args.quiet)
    else:
        subset = 'training' if args.training else 'evaluation'
        results = solve_all_tasks(subset, verbose=not args.quiet)
    
    if not args.quiet:
        print_summary(results)


if __name__ == '__main__':
    main()
