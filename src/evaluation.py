"""
Evaluation harness for testing the Context Thread Agent
"""

from typing import List, Dict
from pathlib import Path
from src.parser import NotebookParser
from src.dependencies import ContextThreadBuilder
from src.indexing import FAISSIndexer
from src.retrieval import RetrievalEngine
from src.reasoning import ContextualAnsweringSystem
from src.intent import ContextThreadEnricher


class EvaluationHarness:
    """Evaluation harness for notebook Q&A."""
    
    def __init__(self, notebooks_dir: str):
        self.notebooks_dir = Path(notebooks_dir)
        self.results = []
    
    def evaluate_all(self, queries_per_notebook: int = 3) -> Dict:
        """Evaluate all notebooks in the directory."""
        notebook_files = list(self.notebooks_dir.glob("*.ipynb"))
        
        for nb_file in notebook_files:
            print(f"Evaluating {nb_file.name}...")
            result = self.evaluate_notebook(str(nb_file), queries_per_notebook)
            self.results.append(result)
        
        return self._aggregate_results()
    
    def evaluate_notebook(self, notebook_path: str, num_queries: int) -> Dict:
        """Evaluate a single notebook."""
        # Load notebook
        parser = NotebookParser()
        result = parser.parse_file(notebook_path)
        cells = result['cells']
        
        # Build context thread
        builder = ContextThreadBuilder(
            notebook_name=Path(notebook_path).stem,
            thread_id=f"eval_{id(self)}"
        )
        builder.add_cells(cells)
        thread = builder.build()
        
        # Enrich
        enricher = ContextThreadEnricher()
        thread = enricher.enrich(thread)
        
        # Index
        indexer = FAISSIndexer()
        indexer.add_multiple(thread.units)
        
        # Setup systems
        engine = RetrievalEngine(thread, indexer)
        answering_system = ContextualAnsweringSystem(engine)
        
        # Generate sample queries (simplified)
        queries = self._generate_sample_queries(cells, num_queries)
        
        # Evaluate
        scores = []
        for query in queries:
            try:
                response = answering_system.answer_question(query)
                # Simple scoring based on citations
                score = min(len(response.citations) * 0.2, 1.0)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        
        return {
            'notebook': Path(notebook_path).name,
            'total_cells': len(cells),
            'queries_tested': len(queries),
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'scores': scores
        }
    
    def _generate_sample_queries(self, cells: List, num_queries: int) -> List[str]:
        """Generate sample queries for evaluation."""
        queries = []
        
        # Simple query generation
        if any('Q4' in cell.source for cell in cells):
            queries.append("Why did we remove Q4 data?")
        
        if any('plot' in cell.source.lower() for cell in cells):
            queries.append("What does the visualization show?")
        
        if any('model' in cell.source.lower() for cell in cells):
            queries.append("What model was used?")
        
        # Fill with generic queries
        generic_queries = [
            "What is the main purpose of this analysis?",
            "What data was used?",
            "What were the key findings?"
        ]
        
        while len(queries) < num_queries and generic_queries:
            queries.append(generic_queries.pop(0))
        
        return queries[:num_queries]
    
    def _aggregate_results(self) -> Dict:
        """Aggregate results across all notebooks."""
        if not self.results:
            return {}
        
        total_notebooks = len(self.results)
        avg_score = sum(r['avg_score'] for r in self.results) / total_notebooks
        total_queries = sum(r['queries_tested'] for r in self.results)
        
        return {
            'total_notebooks': total_notebooks,
            'total_queries': total_queries,
            'average_score': avg_score,
            'results': self.results
        }
    
    def print_summary(self, summary: Dict):
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Notebooks evaluated: {summary.get('total_notebooks', 0)}")
        print(f"Total queries: {summary.get('total_queries', 0)}")
        print(".2%")
        
        for result in summary.get('results', []):
            print(f"\n{result['notebook']}:")
            print(f"  Cells: {result['total_cells']}")
            print(f"  Queries: {result['queries_tested']}")
            print(".2%")
    
    def save_results(self, output_file: str):
        """Save results to CSV."""
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['notebook', 'total_cells', 'queries_tested', 'avg_score'])
            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'notebook': result['notebook'],
                    'total_cells': result['total_cells'],
                    'queries_tested': result['queries_tested'],
                    'avg_score': result['avg_score']
                })