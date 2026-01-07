"""
Context thread building and dependency analysis
"""

from typing import List, Dict, Set, Optional
import re
from src.models import Cell, ContextUnit, ContextThread, CellType


class ContextThreadBuilder:
    """Builds context threads from notebook cells."""
    
    def __init__(self, notebook_name: str, thread_id: str):
        self.notebook_name = notebook_name
        self.thread_id = thread_id
        self.cells: List[Cell] = []
    
    def add_cells(self, cells: List[Cell]):
        """Add cells to the thread."""
        self.cells.extend(cells)
    
    def build(self) -> ContextThread:
        """Build the context thread with dependencies."""
        units = []
        
        # Create units for each cell
        for cell in self.cells:
            dependencies = self._analyze_dependencies(cell, self.cells[:units.index(cell) if cell in units else len(units)])
            unit = ContextUnit(
                cell=cell,
                intent="[Pending intent inference]",
                dependencies=dependencies
            )
            units.append(unit)
        
        # Analyze for cycles
        cycles_detected = self._detect_cycles(units)
        
        metadata = {
            'cycles_detected': cycles_detected,
            'total_cells': len(units)
        }
        
        return ContextThread(
            notebook_name=self.notebook_name,
            thread_id=self.thread_id,
            units=units,
            metadata=metadata
        )
    
    def _analyze_dependencies(self, cell: Cell, previous_cells: List[Cell]) -> List[str]:
        """Analyze dependencies for a cell."""
        dependencies = []
        
        if cell.cell_type != CellType.CODE:
            return dependencies
        
        # Simple heuristic: look for variable assignments and usages
        defined_vars = set()
        used_vars = set()
        
        # Extract variable names from source
        var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        matches = re.findall(var_pattern, cell.source)
        
        # For simplicity, assume all variables in code cells are potential dependencies
        # In a real implementation, you'd use AST parsing
        for var in matches:
            if var not in ['print', 'import', 'from', 'def', 'class', 'if', 'for', 'while']:
                used_vars.add(var)
        
        # Check previous cells for variable definitions
        for prev_cell in previous_cells:
            if prev_cell.cell_type == CellType.CODE:
                prev_matches = re.findall(var_pattern, prev_cell.source)
                for var in prev_matches:
                    if '=' in prev_cell.source and var in used_vars:
                        dependencies.append(prev_cell.cell_id)
                        break
        
        return list(set(dependencies))
    
    def _detect_cycles(self, units: List[ContextUnit]) -> int:
        """Detect cycles in dependencies (simplified)."""
        # For now, return 0 (no cycle detection implemented)
        return 0