#!/usr/bin/env python3
"""
Main entry point for Context Thread Agent.
Handles CLI commands: run-ui, download-notebooks, evaluate
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Context Thread Agent - Notebook Copilot"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch Gradio UI")
    ui_parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    ui_parser.add_argument("--share", action="store_true", help="Create public share link")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download sample notebooks")
    download_parser.add_argument(
        "--output", 
        default="data/sample_notebooks",
        help="Output directory"
    )
    download_parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Target number of notebooks"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation harness")
    eval_parser.add_argument(
        "--notebooks",
        default="data/sample_notebooks",
        help="Directory with notebooks to evaluate"
    )
    eval_parser.add_argument(
        "--queries-per",
        type=int,
        default=3,
        help="Queries per notebook"
    )
    eval_parser.add_argument(
        "--output",
        default="data/evaluation_results.csv",
        help="Output CSV file"
    )
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo script")
    
    args = parser.parse_args()
    
    if args.command == "ui":
        print(f"🚀 Launching UI on port {args.port}...\n")
        from ui.app import create_gradio_app
        
        demo = create_gradio_app()
        demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share
        )
    
    elif args.command == "download":
        print(f"📥 Downloading notebooks to {args.output}...\n")
        from src.notebook_downloader import NotebookDownloader
        
        downloader = NotebookDownloader(args.output)
        notebooks = downloader.download_all()
        
        print(f"\n✅ Downloaded {len(notebooks)} notebooks")
        print(f"   Location: {args.output}")
    
    elif args.command == "evaluate":
        print(f"📊 Evaluating notebooks...\n")
        from src.evaluation import EvaluationHarness
        
        harness = EvaluationHarness(args.notebooks)
        summary = harness.evaluate_all(args.queries_per)
        harness.print_summary(summary)
        harness.save_results(args.output)
    
    elif args.command == "demo":
        print("🎬 Running demo...\n")
        exec(open("demo.py").read())
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()