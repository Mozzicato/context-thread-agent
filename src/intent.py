"""
Intent inference for notebook cells
"""

from typing import List, Optional
import re
from src.models import ContextUnit, CellType


class ContextThreadEnricher:
    """Enriches context threads with intent information."""
    
    def __init__(self, infer_intents: bool = True):
        self.infer_intents = infer_intents
    
    def enrich(self, thread):
        """Enrich the thread with intents."""
        if not self.infer_intents:
            return thread
        
        for unit in thread.units:
            if unit.intent == "[Pending intent inference]":
                unit.intent = self._infer_intent_heuristic(unit)
        
        return thread
    
    def _infer_intent_heuristic(self, unit: ContextUnit) -> str:
        """Infer intent using heuristics."""
        source = unit.cell.source.lower()
        
        # Data loading
        if any(keyword in source for keyword in ['read_csv', 'read_excel', 'load', 'open']):
            return "Load data from file"
        
        # Data cleaning
        if any(keyword in source for keyword in ['dropna', 'fillna', 'clean', 'remove', 'filter']):
            return "Clean and preprocess data"
        
        # Analysis/Modeling
        if any(keyword in source for keyword in ['fit', 'predict', 'train', 'model', 'regression']):
            return "Build and train model"
        
        # Visualization
        if any(keyword in source for keyword in ['plot', 'chart', 'graph', 'visualize', 'show']):
            return "Create visualization"
        
        # Statistics
        if any(keyword in source for keyword in ['mean', 'std', 'sum', 'count', 'describe']):
            return "Compute statistics"
        
        # Default
        return "Execute code"