"""
Data models for Context Thread Agent
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


class CellType(Enum):
    CODE = "code"
    MARKDOWN = "markdown"
    RAW = "raw"


@dataclass
class Cell:
    """Represents a Jupyter notebook cell."""
    cell_id: str
    cell_type: CellType
    source: str
    outputs: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.outputs is None:
            self.outputs = []


@dataclass
class ContextUnit:
    """A cell with its context and dependencies."""
    cell: Cell
    intent: str
    dependencies: List[str]
    context_window: List[str] = None
    
    def __post_init__(self):
        if self.context_window is None:
            self.context_window = []


@dataclass
class ContextThread:
    """A thread of related context units."""
    notebook_name: str
    thread_id: str
    units: List[ContextUnit]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryRequest:
    """A user query request."""
    query: str
    notebook_path: Optional[str] = None
    top_k: int = 5


@dataclass
class Citation:
    """A citation to a specific cell."""
    cell_id: str
    cell_type: CellType
    content_snippet: str
    intent: Optional[str] = None


@dataclass
class AgentResponse:
    """Response from the agent."""
    answer: str
    citations: List[Citation]
    confidence: float
    has_hallucination_risk: bool
    retrieved_units: List[ContextUnit]


@dataclass
class RetrievalResult:
    """Result from retrieval system."""
    units: List[ContextUnit]
    scores: List[float]
    query: str