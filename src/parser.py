"""
Jupyter notebook parsing utilities
"""

import json
from typing import Dict, List, Any
from pathlib import Path
from src.models import Cell, CellType


class NotebookParser:
    """Parser for Jupyter notebooks."""
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a Jupyter notebook file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Notebook file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            nb_data = json.load(f)
        
        cells = []
        for i, cell_data in enumerate(nb_data.get('cells', [])):
            cell = self._parse_cell(cell_data, i)
            cells.append(cell)
        
        return {
            'cells': cells,
            'metadata': nb_data.get('metadata', {}),
            'nbformat': nb_data.get('nbformat', 4)
        }
    
    def _parse_cell(self, cell_data: Dict[str, Any], index: int) -> Cell:
        """Parse a single cell."""
        cell_type_str = cell_data.get('cell_type', 'code')
        cell_type = CellType(cell_type_str)
        
        source = ''.join(cell_data.get('source', []))
        
        # Generate cell ID
        cell_id = f"cell_{index}"
        
        # For code cells, include outputs
        outputs = []
        if cell_type == CellType.CODE:
            outputs = cell_data.get('outputs', [])
        
        return Cell(
            cell_id=cell_id,
            cell_type=cell_type,
            source=source,
            outputs=outputs
        )