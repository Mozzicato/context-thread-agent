"""
Groq LLM Integration for Context Thread Agent
Provides fast, free reasoning without OpenAI API costs
"""

import os
from typing import Optional
import json
from groq import Groq


class GroqReasoningEngine:
    """
    Alternative reasoning engine using Groq API
    Faster and free compared to OpenAI
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq client"""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key or self.api_key == "YOUR_GROQ_API_KEY_HERE":
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"  # Latest Groq model
        
    def reason_with_context(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[list] = None
    ) -> dict:
        """
        Generate answer using Groq with context and conversation history
        Returns: {answer, confidence, citations}
        """
        if system_prompt is None:
            system_prompt = """You are an expert data science assistant analyzing Jupyter notebooks and Excel documents.
Answer ONLY based on the provided context. Do not make up or infer information beyond what's shown.
If asked something not in the context, say "I cannot find that information in the provided document".

When referencing specific parts of code or analysis, cite them clearly using [Cell X] format.
Be precise, detailed, and technical in your answers.
If you see patterns, trends, or issues in the data/code, highlight them."""
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add conversation history for context continuity
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 3 exchanges
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current query with context
        messages.append({
            "role": "user",
            "content": f"Context from the document:\n{context}\n\nQuestion: {question}\n\nProvide a detailed answer based on the context:"
        })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.2,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "confidence": 0.85,
                "citations": []
            }
        except Exception as e:
            return {
                "answer": f"Error using Groq: {str(e)}",
                "confidence": 0.0,
                "citations": []
            }
    
    def generate_keypoints(
        self,
        context: str,
        max_points: int = 10
    ) -> dict:
        """
        Generate key insights and summary points from document
        """
        system_prompt = """You are an expert at analyzing data science notebooks and Excel documents.
Generate a comprehensive summary with key insights, findings, and important points.
Focus on: methodology, data transformations, key findings, issues/concerns, and conclusions.
Format your response as a clear, bulleted list."""
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""Analyze this document and provide {max_points} key points covering:
1. Purpose and methodology
2. Data characteristics and transformations
3. Key findings or patterns
4. Any issues, concerns, or anomalies
5. Overall conclusions

Document context:
{context}

Provide your analysis:"""
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2500,
                temperature=0.3
            )
            
            keypoints = response.choices[0].message.content
            
            return {
                "keypoints": keypoints,
                "success": True
            }
        except Exception as e:
            return {
                "keypoints": f"Error generating keypoints: {str(e)}",
                "success": False
            }