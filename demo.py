#!/usr/bin/env python3
"""
Demo script for Context Thread Agent
Loads the demo notebook and runs sample queries
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parser import NotebookParser
from src.dependencies import ContextThreadBuilder
from src.indexing import FAISSIndexer
from src.retrieval import RetrievalEngine, ContextBuilder
from src.reasoning import ContextualAnsweringSystem
from src.intent import ContextThreadEnricher
from src.groq_integration import GroqReasoningEngine


def run_demo():
    """Run the demo with sample queries."""
    print("🎬 Context Thread Agent Demo")
    print("=" * 50)
    
    # Load demo notebook
    print("\n📓 Loading demo notebook...")
    parser = NotebookParser()
    result = parser.parse_file("data/demo_notebook.ipynb")
    cells = result['cells']
    
    # Build context thread
    builder = ContextThreadBuilder(
        notebook_name="demo_notebook",
        thread_id="demo_thread"
    )
    builder.add_cells(cells)
    thread = builder.build()
    
    # Enrich with intents
    enricher = ContextThreadEnricher(infer_intents=True)
    thread = enricher.enrich(thread)
    
    # Index
    indexer = FAISSIndexer()
    indexer.add_multiple(thread.units)
    
    # Setup retrieval and reasoning
    engine = RetrievalEngine(thread, indexer)
    answering_system = ContextualAnsweringSystem(engine)
    
    print(f"✅ Loaded {len(cells)} cells, indexed {len(thread.units)} units")
    
    # Sample queries
    queries = [
        "Why did we remove Q4 data?",
        "What transformations were applied to the data?",
        "What is the main insight from this analysis?"
    ]
    
    print("\n❓ Running sample queries...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 40)
        
        try:
            response = answering_system.answer_question(query, top_k=5)
            
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence:.1%}")
            
            if response.citations:
                print(f"Citations: {len(response.citations)} cells referenced")
            
            print()
            
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("🎉 Demo complete!")


if __name__ == "__main__":
    run_demo()