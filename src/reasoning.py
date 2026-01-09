"""
LLM reasoning layer for answering questions with citations.
Ensures all responses are grounded in retrieved context.
"""

import os
import re
from typing import List, Dict, Optional

from src.models import QueryRequest, AgentResponse, Citation, ContextUnit
from src.retrieval import RetrievalResult, ContextBuilder
from src.groq_integration import GroqReasoningEngine


class CitationExtractor:
    """Extract cell references from LLM responses."""
    
    @staticmethod
    def extract_citations(response_text: str, retrieved_units: List[ContextUnit]) -> List[Citation]:
        """
        Extract cell citations from LLM response.
        Looks for patterns like "Cell X", "cell_X", "(X)", etc.
        """
        citations = []
        cell_ids = {u.cell.cell_id for u in retrieved_units}
        
        # Pattern 1: "Cell X" or "cell X"
        pattern1 = r'[Cc]ell\s+([a-zA-Z_0-9]+)'
        matches1 = re.findall(pattern1, response_text)
        
        # Pattern 2: "(cell_X)" or similar
        pattern2 = r'\(([a-zA-Z_0-9]+)\)'
        matches2 = re.findall(pattern2, response_text)
        
        # Combine and deduplicate
        potential_cells = set(matches1 + matches2)
        
        # Validate against retrieved cells
        for cell_id in potential_cells:
            if cell_id in cell_ids:
                # Find the unit
                unit = next((u for u in retrieved_units if u.cell.cell_id == cell_id), None)
                if unit:
                    citation = Citation(
                        cell_id=cell_id,
                        cell_type=unit.cell.cell_type,
                        content_snippet=CitationExtractor._get_snippet(unit),
                        intent=unit.intent if unit.intent != "[Pending intent inference]" else None
                    )
                    citations.append(citation)
        
        return citations
    
    @staticmethod
    def _get_snippet(unit: ContextUnit, max_length: int = 100) -> str:
        """Get content snippet from unit."""
        return unit.cell.source[:max_length]


class HallucinationDetector:
    """Detect potential hallucinations in responses."""
    
    @staticmethod
    def check_for_unsupported_claims(response: str, context: str) -> bool:
        """Check if response makes claims not supported by context."""
        # Simplified check - in real implementation, use more sophisticated methods
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Check for common hallucination indicators
        hallucination_indicators = [
            "according to", "experts say", "research shows",
            "it's known that", "generally", "typically"
        ]
        
        for indicator in hallucination_indicators:
            if indicator in response_lower and indicator not in context_lower:
                return True
        
        return False


class ReasoningEngine:
    """LLM reasoning engine for answering questions."""
    
    def __init__(self):
        self.groq_client = self._init_groq()
        self.openai_client = self._init_openai()
    
    def _init_groq(self):
        """Initialize Groq client (preferred for speed and cost)."""
        try:
            return GroqReasoningEngine()
        except Exception:
            return None
    
    def _init_openai(self):
        """Initialize OpenAI client (fallback)."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and not api_key.startswith("sk-placeholder"):
                from openai import OpenAI
                return OpenAI(api_key=api_key)
        except Exception:
            pass
        return None
    
    def reason(self, query: str, retrieval_result: RetrievalResult, conversation_history: Optional[List[Dict]] = None) -> AgentResponse:
        """
        Reason about a question given retrieved context.
        Returns response with citations.
        """
        # Build context for LLM
        context = ContextBuilder.build_context_for_llm(
            retrieval_result.units,
            query
        )
        
        # Try to use Groq first (fast and free)
        if self.groq_client:
            try:
                groq_result = self.groq_client.reason_with_context(query, context, conversation_history=conversation_history)
                answer = groq_result["answer"]
            except Exception as e:
                print(f"Groq query failed: {e}. Using OpenAI fallback.")
                answer = self._query_openai(query, context) if self.openai_client else self._generate_answer_fallback(query, retrieval_result)
        elif self.openai_client:
            try:
                answer = self._query_openai(query, context)
            except Exception as e:
                print(f"OpenAI query failed: {e}. Using fallback.")
                answer = self._generate_answer_fallback(query, retrieval_result)
        else:
            # Use fallback reasoning
            answer = self._generate_answer_fallback(query, retrieval_result)
        
        # Extract citations
        citations = CitationExtractor.extract_citations(answer, retrieval_result.units)
        
        # Check for hallucination risk
        has_hallucination_risk = HallucinationDetector.check_for_unsupported_claims(
            answer, context
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            len(citations),
            len(retrieval_result.units),
        )
        
        return AgentResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            has_hallucination_risk=has_hallucination_risk,
            retrieved_units=retrieval_result.units
        )
    
    def _query_openai(self, query: str, context: str) -> str:
        """Query OpenAI API."""
        if not self.openai_client:
            raise Exception("OpenAI client not available")
        
        prompt = f"""
Based on the following notebook context, answer the question. 
Cite specific cells when referencing information.

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def _generate_answer_fallback(self, query: str, retrieval_result: RetrievalResult) -> str:
        """Generate answer using simple fallback logic."""
        query_lower = query.lower()
        
        # Handle specific question types
        # Match: "what is this notebook about", "whats the notebook about", "what's this about", etc.
        if any(phrase in query_lower for phrase in [
            "what is this notebook about", "what does this notebook", 
            "whats the notebook", "what's this", "what about this notebook",
            "what is this about", "describe this notebook", "tell me about this"
        ]):
            return self._summarize_notebook(retrieval_result)
        
        if "why" in query_lower:
            return self._explain_decision(retrieval_result, query)
        
        # Default: find relevant code snippets and summarize all units
        if retrieval_result.units:
            return self._summarize_notebook(retrieval_result)
        
        return "I couldn't find specific information about that in the notebook context."
    
    def _summarize_notebook(self, retrieval_result: RetrievalResult) -> str:
        """Generate a comprehensive summary of what the notebook is about."""
        data_sources = []
        data_operations = []
        models = []
        metrics = []
        visualizations = []
        code_cells = 0
        markdown_cells = 0
        
        for unit in retrieval_result.units:
            intent = unit.intent.lower() if unit.intent else ""
            source = unit.cell.source.lower()
            
            if unit.cell.cell_type == "code":
                code_cells += 1
            elif unit.cell.cell_type == "markdown":
                markdown_cells += 1
            
            # Data loading/sources
            if "load data" in intent or "read" in source or "dataset" in source:
                if "iris" in source:
                    data_sources.append("Iris dataset")
                elif "csv" in source or "pd.read_csv" in source:
                    data_sources.append("CSV data files")
                elif "excel" in source or "xlsx" in source:
                    data_sources.append("Excel spreadsheets")
                else:
                    data_sources.append("external datasets")
            
            # Data operations
            if "preprocess" in intent or "clean" in source or "drop" in source:
                data_operations.append("data cleaning and preprocessing")
            if "filter" in source or "select" in source:
                data_operations.append("data filtering")
            if "merge" in source or "join" in source:
                data_operations.append("data merging")
            
            # Models
            if "model" in intent or "fit" in source or "train" in source:
                if "randomforest" in source:
                    models.append("Random Forest classifier")
                elif "regression" in source:
                    models.append("regression model")
                elif "neural" in source or "nn" in source:
                    models.append("neural network")
                else:
                    models.append("machine learning model")
            
            # Evaluation metrics
            if "accuracy" in source or "precision" in source or "recall" in source or "f1" in source:
                metrics.append("classification metrics")
            if "rmse" in source or "mse" in source:
                metrics.append("regression metrics")
            if "auc" in source or "roc" in source:
                metrics.append("ROC/AUC analysis")
            
            # Visualizations
            if "visualize" in intent or "plot" in source or "matplotlib" in source or "seaborn" in source:
                if "scatter" in source:
                    visualizations.append("scatter plots")
                elif "hist" in source:
                    visualizations.append("histograms")
                elif "bar" in source:
                    visualizations.append("bar charts")
                else:
                    visualizations.append("data visualizations")
        
        # Build comprehensive summary
        summary = []
        
        # Main purpose
        if data_sources and models:
            summary.append(f"This is a machine learning notebook that analyzes {', '.join(set(data_sources))}")
        elif data_sources:
            summary.append(f"This notebook analyzes {', '.join(set(data_sources))}")
        elif models:
            summary.append("This notebook demonstrates machine learning model development and evaluation")
        else:
            summary.append("This is a data analysis notebook")
        
        # Data operations
        if data_operations:
            summary.append(f"It includes {', '.join(set(data_operations))}")
        
        # Models and evaluation
        if models or metrics:
            model_desc = f"Uses {', '.join(set(models))}" if models else "Includes model training"
            if metrics:
                model_desc += f" with {', '.join(set(metrics))}"
            summary.append(model_desc)
        
        # Visualizations
        if visualizations:
            summary.append(f"Includes {', '.join(set(visualizations))} for data exploration and results visualization")
        
        # Notebook structure
        total_cells = code_cells + markdown_cells
        if total_cells > 0:
            summary.append(f"\n**Notebook Structure:** {code_cells} code cells, {markdown_cells} documentation cells")
        
        return ". ".join(summary) + "."
    
    def _explain_decision(self, retrieval_result: RetrievalResult, query: str) -> str:
        """Explain why certain decisions were made."""
        query_lower = query.lower()
        
        # Look for common decisions
        if "remove" in query_lower or "drop" in query_lower:
            for unit in retrieval_result.units:
                if "drop" in unit.cell.source.lower() or "remove" in unit.cell.source.lower():
                    return f"Data was removed/cleaned as shown in: {unit.cell.source[:150]}"
        
        return "The notebook shows standard data preprocessing and modeling steps."
    
    def _calculate_confidence(self, num_citations: int, num_units: int) -> float:
        """Calculate confidence score."""
        if num_units == 0:
            return 0.0
        
        # If we have units but no explicit citations, give baseline confidence (0.7)
        if num_citations == 0 and num_units > 0:
            return 0.7
        
        # With citations, confidence increases
        base_confidence = min(num_citations / max(num_units, 1), 1.0)
        return max(base_confidence, 0.7)


class ContextualAnsweringSystem:
    """End-to-end system for context-aware question answering."""
    
    def __init__(self, retrieval_engine, use_llm: bool = True):
        self.retrieval_engine = retrieval_engine
        self.reasoning_engine = ReasoningEngine()
        self.use_llm = use_llm
    
    def answer_question(self, query: str, top_k: int = 5, conversation_history: Optional[List[Dict]] = None) -> AgentResponse:
        """
        Answer a question about the notebook context.
        
        Args:
            query: User's natural language question
            top_k: Number of cells to retrieve for context
            conversation_history: Previous conversation for context
            
        Returns:
            AgentResponse with answer, citations, and context
        """
        # Step 1: Retrieve relevant context
        retrieval_result = self.retrieval_engine.retrieve(query, top_k=top_k)
        
        # Step 2: Reason and generate answer with conversation context
        response = self.reasoning_engine.reason(query, retrieval_result, conversation_history)
        
        return response