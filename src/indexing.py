"""
FAISS-based indexing for context units
"""

import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from src.models import ContextUnit


class FAISSIndexer:
    """FAISS-based vector indexer for context units."""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = None
        self.context_units: Dict[str, ContextUnit] = {}
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        except ImportError:
            print("FAISS not installed, using dummy index")
            self.index = None
    
    def add_context_unit(self, unit: ContextUnit):
        """Add a single context unit."""
        if self.index is None:
            return
        
        # Generate embedding (simplified - in real implementation use actual embeddings)
        embedding = self._get_embedding(unit)
        
        # Add to index
        self.index.add(np.array([embedding], dtype=np.float32))
        self.context_units[unit.cell.cell_id] = unit
    
    def add_multiple(self, units: List[ContextUnit]):
        """Add multiple context units."""
        for unit in units:
            self.add_context_unit(unit)
    
    def search_units(self, query: str, k: int = 5) -> List[Tuple[ContextUnit, float]]:
        """Search for similar units."""
        if self.index is None or len(self.context_units) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self._get_embedding_from_text(query)
        
        # Search
        scores, indices = self.index.search(np.array([query_embedding], dtype=np.float32), min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.context_units):
                unit_id = list(self.context_units.keys())[idx]
                unit = self.context_units[unit_id]
                results.append((unit, float(score)))
        
        return results
    
    def _get_embedding(self, unit: ContextUnit) -> np.ndarray:
        """Get embedding for a context unit."""
        text = f"{unit.intent} {unit.cell.source}"
        return self._get_embedding_from_text(text)
    
    def _get_embedding_from_text(self, text: str) -> np.ndarray:
        """Get embedding from text (simplified)."""
        # In real implementation, use OpenAI or other embedding API
        # For now, return random vector
        np.random.seed(hash(text) % 2**32)
        return np.random.normal(0, 1, self.dimension).astype(np.float32)