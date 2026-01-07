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
        if "what is this notebook about" in query_lower or "what does this notebook do" in query_lower:
            return self._summarize_notebook(retrieval_result)
        
        if "why" in query_lower:
            return self._explain_decision(retrieval_result, query)
        
        # Default: find relevant code snippets
        relevant_snippets = []
        for unit in retrieval_result.units:
            source_lower = unit.cell.source.lower()
            if any(word in query_lower.split() for word in ['what', 'how', 'why', 'when', 'where']):
                relevant_snippets.append(unit.cell.source[:100])
        
        if relevant_snippets:
            return f"Based on the notebook content: {'; '.join(relevant_snippets[:3])}"
        
        return "I couldn't find specific information about that in the notebook context."
    
    def _summarize_notebook(self, retrieval_result: RetrievalResult) -> str:
        """Generate a summary of what the notebook is about."""
        intents = []
        data_sources = []
        models = []
        visualizations = []
        
        for unit in retrieval_result.units:
            intent = unit.intent.lower() if unit.intent else ""
            source = unit.cell.source.lower()
            
            if "load data" in intent or "read" in source:
                if "iris" in source:
                    data_sources.append("Iris dataset")
                elif "csv" in source:
                    data_sources.append("CSV file")
                else:
                    data_sources.append("external data")
            
            if "model" in intent or "fit" in source or "train" in source:
                if "randomforest" in source:
                    models.append("Random Forest classifier")
                elif "regression" in source:
                    models.append("regression model")
                else:
                    models.append("machine learning model")
            
            if "visualize" in intent or "plot" in source:
                visualizations.append("data visualization")
        
        summary_parts = []
        if data_sources:
            summary_parts.append(f"This notebook analyzes {', '.join(set(data_sources))}")
        if models:
            summary_parts.append(f"using {', '.join(set(models))}")
        if visualizations:
            summary_parts.append("and creates visualizations")
        
        if summary_parts:
            return " ".join(summary_parts) + "."
        else:
            return "This appears to be a data analysis notebook with machine learning components."
    
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
        
        # Simple formula: more citations = higher confidence
        base_confidence = min(num_citations / max(num_units, 1), 1.0)
        return base_confidence


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