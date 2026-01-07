"""
Retrieval system for context-aware question answering
"""

from typing import List, Dict, Optional
from src.models import ContextUnit, RetrievalResult, QueryRequest


class ContextBuilder:
    """Builds context windows for LLM queries."""
    
    @staticmethod
    def build_context_for_llm(units: List[ContextUnit], query: str, max_tokens: int = 3000) -> str:
        """Build context string for LLM."""
        context_parts = []
        
        for unit in units:
            part = f"Cell {unit.cell.cell_id} ({unit.cell.cell_type}):\n"
            if unit.intent and unit.intent != "[Pending intent inference]":
                part += f"Intent: {unit.intent}\n"
            part += f"Content: {unit.cell.source[:500]}\n"
            if unit.dependencies:
                part += f"Dependencies: {', '.join(unit.dependencies)}\n"
            part += "\n"
            
            context_parts.append(part)
        
        context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > max_tokens * 4:  # Rough token estimate
            context = context[:max_tokens * 4] + "..."
        
        return context


class RetrievalEngine:
    """Main retrieval engine."""
    
    def __init__(self, context_thread, indexer):
        self.context_thread = context_thread
        self.indexer = indexer
    
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve relevant context units."""
        # Use semantic search
        semantic_results = self.indexer.search_units(query, k=top_k)
        
        # Extract units and scores
        units = [unit for unit, score in semantic_results]
        scores = [score for unit, score in semantic_results]
        
        # For now, just return semantic results
        # In full implementation, combine with structural retrieval
        
        return RetrievalResult(
            units=units,
            scores=scores,
            query=query
        )