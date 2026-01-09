"""
Gradio UI for Context Thread Agent - Enterprise Edition
Professional document analysis with killer features
"""

import gradio as gr
import json
import tempfile
import os
import html
from pathlib import Path
from typing import Tuple, List, Dict
from src.models import Cell, CellType
from datetime import datetime

from src.parser import NotebookParser
from src.dependencies import ContextThreadBuilder
from src.indexing import FAISSIndexer
from src.retrieval import RetrievalEngine, ContextBuilder
from src.reasoning import ContextualAnsweringSystem
from src.intent import ContextThreadEnricher
from src.groq_integration import GroqReasoningEngine
import pandas as pd


class NotebookAgentUI:
    """Enterprise-grade Gradio UI for the Context Thread Agent."""
    
    def __init__(self):
        self.current_thread = None
        self.current_indexer = None
        self.current_engine = None
        self.answering_system = None
        self.conversation_history = []
        self.groq_client = None
        self.keypoints_generated = False
        self.keypoints_cache = None
        self.current_file_name = None
        self.data_profile = None
        self.current_file_path = None
        self.current_file_ext = None
        
        # Initialize Groq client
        try:
            self.groq_client = GroqReasoningEngine()
        except Exception as e:
            print(f"Warning: Groq not initialized: {e}")
    
    def load_notebook(self, notebook_file) -> Tuple[str, bool, str, str]:
        """Load and index a notebook or Excel file."""
        try:
            if notebook_file is None:
                return "❌ No file provided", False, "", ""
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix=Path(notebook_file).suffix if isinstance(notebook_file, str) else ".ipynb", delete=False) as f:
                if isinstance(notebook_file, str):
                    f.write(open(notebook_file, 'rb').read())
                else:
                    f.write(notebook_file.read())
                temp_path = f.name
            
            file_ext = Path(temp_path).suffix.lower()
            
            if file_ext == '.ipynb':
                parser = NotebookParser()
                result = parser.parse_file(temp_path)
                cells = result['cells']
            elif file_ext in ['.xlsx', '.xls']:
                cells = self._excel_to_cells(temp_path)
            else:
                return "❌ Unsupported file type. Please upload .ipynb or .xlsx/.xls", False, "", ""
            
            # Build context thread
            builder = ContextThreadBuilder(
                notebook_name=Path(temp_path).stem,
                thread_id=f"thread_{id(self)}"
            )
            builder.add_cells(cells)
            self.current_thread = builder.build()
            
            # Enrich with intents
            enricher = ContextThreadEnricher(infer_intents=True)
            self.current_thread = enricher.enrich(self.current_thread)
            
            # Index
            self.current_indexer = FAISSIndexer()
            self.current_indexer.add_multiple(self.current_thread.units)
            
            # Setup retrieval and reasoning
            self.current_engine = RetrievalEngine(self.current_thread, self.current_indexer)
            self.answering_system = ContextualAnsweringSystem(self.current_engine)
            
            # Reset conversation
            self.conversation_history = []
            self.keypoints_generated = False
            self.keypoints_cache = None
            
            # Store file info for later use
            self.current_file_path = temp_path
            self.current_file_ext = file_ext
            
            # Get appropriate preview based on file type
            if file_ext in ['.xlsx', '.xls']:
                notebook_preview = self.get_excel_display(temp_path)
            else:
                notebook_preview = self.get_notebook_display()
                # Cleanup for non-Excel files
                Path(temp_path).unlink()
            
            status_msg = f"""
### ✅ File Loaded Successfully!

**Document Statistics:**
- Total sections: {len(cells)}
- Code sections: {sum(1 for c in cells if c.cell_type == CellType.CODE)}
- Documentation: {sum(1 for c in cells if c.cell_type == CellType.MARKDOWN)}
- Indexed & Ready: ✓

You can now:
- 🔍 Browse the document in the viewer
- 🔑 Generate key insights (recommended)
- ❓ Ask any questions about the content
"""
            
            return status_msg, True, notebook_preview, ""
        
        except Exception as e:
            return f"❌ Error loading file: {str(e)}", False, "", ""
    
    def generate_keypoints(self) -> str:
        """Generate key points summary using Groq."""
        if not self.answering_system:
            return "❌ No document loaded."
        
        if self.keypoints_cache:
            return self.keypoints_cache
        
        try:
            # Get comprehensive context
            all_context = []
            for unit in self.current_thread.units[:30]:  # First 30 cells
                all_context.append(f"### {unit.cell.cell_id} [{unit.cell.cell_type}]")
                if unit.intent and unit.intent != "[Pending intent inference]":
                    all_context.append(f"Intent: {unit.intent}")
                source_text = unit.cell.source if isinstance(unit.cell.source, str) else ''.join(unit.cell.source)
                all_context.append(source_text[:500])
                if unit.cell.outputs:
                    for output in unit.cell.outputs[:1]:
                        if 'text' in output:
                            raw_out = output['text']
                            if isinstance(raw_out, list):
                                raw_out = '\n'.join(raw_out)
                            all_context.append(f"Output: {raw_out[:200]}")
                all_context.append("---")
            
            context_text = "\n".join(all_context)
            
            # Use Groq to generate keypoints
            if self.groq_client:
                result = self.groq_client.generate_keypoints(context_text, max_points=12)
                if result["success"]:
                    self.keypoints_cache = f"## 🔑 Key Insights & Summary\n\n{result['keypoints']}"
                    self.keypoints_generated = True
                    return self.keypoints_cache
                else:
                    return f"❌ {result['keypoints']}"
            else:
                return "❌ Groq client not available. Please check your API key."
        
        except Exception as e:
            return f"❌ Error generating keypoints: {str(e)}"
    
    def set_groq_key(self, api_key: str, enable: bool) -> str:
        """Set or clear the Groq API key and reinitialize the Groq client at runtime."""
        try:
            if not enable:
                # Disable Groq usage
                self.groq_client = None
                os.environ.pop("GROQ_API_KEY", None)
                return "✅ Groq disabled. The system will use fallback reasoning."
            
            if not api_key or api_key.strip() == "":
                return "❌ Please provide a valid Groq API key to enable Groq."
            
            # Try to initialize Groq with the provided key
            self.groq_client = GroqReasoningEngine(api_key=api_key.strip())
            os.environ["GROQ_API_KEY"] = api_key.strip()
            return "✅ Groq enabled successfully. Using Groq for reasoning."
        except Exception as e:
            self.groq_client = None
            return f"❌ Could not initialize Groq: {str(e)}"
    
    def get_notebook_display(self) -> str:
        """Get Google Colab-like styled notebook content."""
        if not self.current_thread:
            return "No document loaded."
        
        display = """
<style>
:root {
    --colab-primary: #f59b42;
    --colab-secondary: #e8eaed;
    --colab-text: #202124;
    --colab-border: #dadce0;
}

.colab-container {
    font-family: 'Roboto', 'Helvetica Neue', sans-serif;
    color: var(--colab-text);
    padding: 24px;
    background: white;
}

.colab-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 32px;
    padding: 16px;
    background: linear-gradient(135deg, #f59b42 0%, #f5a962 100%);
    border-radius: 8px;
    color: white;
}

.colab-header h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 500;
}

.colab-header-subtitle {
    color: rgba(255,255,255,0.9);
    font-size: 14px;
    margin-top: 4px;
}

.colab-cell {
    background: white;
    border: 1px solid var(--colab-border);
    border-radius: 4px;
    margin: 16px 0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    overflow: hidden;
}

.colab-cell-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: var(--colab-secondary);
    border-bottom: 1px solid var(--colab-border);
    font-size: 12px;
    font-weight: 500;
    color: #5f6368;
}

.colab-cell-number {
    color: #80868b;
    font-family: 'Courier New', monospace;
    font-weight: bold;
}

.colab-cell-type {
    display: inline-block;
    padding: 2px 8px;
    background: white;
    border: 1px solid var(--colab-border);
    border-radius: 2px;
    font-size: 11px;
    font-weight: 500;
}

.colab-cell-type.code {
    background: #f0f0f0;
    color: #1976d2;
}

.colab-cell-type.markdown {
    background: #f0f0f0;
    color: #d32f2f;
}

.colab-cell-intent {
    display: inline-block;
    padding: 3px 8px;
    background: #e3f2fd;
    color: #1976d2;
    border-radius: 2px;
    font-size: 11px;
    font-weight: 500;
    margin-left: auto;
}

.colab-code {
    background: #282c34;
    color: #abb2bf;
    padding: 16px;
    font-family: 'Courier New', 'Monaco', monospace;
    font-size: 13px;
    line-height: 1.6;
    overflow-x: auto;
    position: relative;
}

/* Ensure <pre> inside code blocks inherits visible color and preserves whitespace */
.colab-code pre {
    color: #abb2bf !important;
    white-space: pre !important;
    margin: 0 !important;
    font-family: inherit !important;
    overflow-x: auto;
}

.colab-code-keyword { color: #c678dd; }
.colab-code-string { color: #98c379; }
.colab-code-number { color: #d19a66; }
.colab-code-function { color: #61afef; }
.colab-code-comment { color: #5c6370; font-style: italic; }

.colab-markdown {
    padding: 16px;
    font-size: 14px;
    line-height: 1.7;
}

.colab-markdown h1 { font-size: 32px; font-weight: 500; margin: 24px 0 16px 0; }
.colab-markdown h2 { font-size: 24px; font-weight: 500; margin: 20px 0 12px 0; }
.colab-markdown h3 { font-size: 20px; font-weight: 500; margin: 16px 0 10px 0; }
.colab-markdown p { margin: 12px 0; }
.colab-markdown ul, .colab-markdown ol { margin: 12px 0; padding-left: 24px; }
.colab-markdown code { 
    background: #f5f5f5; 
    padding: 2px 6px; 
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
}
.colab-markdown pre { 
    background: #f5f5f5; 
    padding: 12px; 
    border-radius: 4px;
    overflow-x: auto;
}

.colab-output {
    background: var(--colab-secondary);
    border-top: 1px solid var(--colab-border);
    padding: 12px 16px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    max-height: 400px;
    overflow-y: auto;
}

.colab-output-label {
    font-weight: 600;
    color: #5f6368;
    font-size: 11px;
    margin-bottom: 8px;
}

.colab-stats {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}

.colab-stat {
    flex: 1;
    min-width: 140px;
    background: white;
    border: 1px solid var(--colab-border);
    padding: 16px;
    border-radius: 4px;
    text-align: center;
}

.colab-stat-value {
    font-size: 24px;
    font-weight: 500;
    color: var(--colab-primary);
}

.colab-stat-label {
    font-size: 12px;
    color: #5f6368;
    margin-top: 8px;
}
</style>

<div class="colab-container">
    <div class="colab-header">
        <div>
            <h1>📓 Notebook Analysis</h1>
            <div class="colab-header-subtitle">Google Colab-style Professional Viewer</div>
        </div>
    </div>
"""
        
        code_cells = sum(1 for u in self.current_thread.units if u.cell.cell_type == CellType.CODE)
        markdown_cells = sum(1 for u in self.current_thread.units if u.cell.cell_type == CellType.MARKDOWN)
        cells_with_output = sum(1 for u in self.current_thread.units if u.cell.outputs)
        
        display += f"""
    <div class="colab-stats">
        <div class="colab-stat">
            <div class="colab-stat-value">{len(self.current_thread.units)}</div>
            <div class="colab-stat-label">Total Cells</div>
        </div>
        <div class="colab-stat">
            <div class="colab-stat-value">{code_cells}</div>
            <div class="colab-stat-label">Code Cells</div>
        </div>
        <div class="colab-stat">
            <div class="colab-stat-value">{markdown_cells}</div>
            <div class="colab-stat-label">Documentation</div>
        </div>
        <div class="colab-stat">
            <div class="colab-stat-value">{cells_with_output}</div>
            <div class="colab-stat-label">With Output</div>
        </div>
    </div>
"""
        
        for i, unit in enumerate(self.current_thread.units, 1):
            cell_type_str = "CODE" if unit.cell.cell_type == CellType.CODE else "MARKDOWN"
            cell_type_class = "code" if unit.cell.cell_type == CellType.CODE else "markdown"
            
            display += f"""
    <div class="colab-cell">
        <div class="colab-cell-header">
            <span class="colab-cell-number">[{i}]</span>
            <span class="colab-cell-type {cell_type_class}">{cell_type_str}</span>
"""
            
            if unit.intent and unit.intent != "[Pending intent inference]":
                display += f'            <span class="colab-cell-intent">{unit.intent}</span>\n'
            
            display += """        </div>
"""
            
            if unit.cell.cell_type == CellType.CODE:
                # Escape HTML special characters and preserve whitespace
                # Handle source as either string or list
                source_text = unit.cell.source if isinstance(unit.cell.source, str) else ''.join(unit.cell.source)
                code = html.escape(source_text)
                display += f'        <div class="colab-code"><pre style="margin: 0; color: #abb2bf; white-space: pre; overflow-x: auto; font-family: \"Courier New\", monospace;">{code}</pre></div>\n'
            else:
                # Handle source as either string or list
                source_text = unit.cell.source if isinstance(unit.cell.source, str) else ''.join(unit.cell.source)
                display += f'        <div class="colab-markdown">{source_text}</div>\n'
            
            if unit.cell.outputs:
                display += '        <div class="colab-output">\n'
                display += '            <div class="colab-output-label">Output</div>\n'
                for output in unit.cell.outputs[:2]:
                    if 'text' in output:
                        raw_out = output['text']
                        if isinstance(raw_out, list):
                            raw_out = '\n'.join(raw_out)
                        output_text = html.escape(str(raw_out)[:300])
                        display += f'            <pre>{output_text}</pre>\n'
                    elif 'data' in output and 'text/plain' in output['data']:
                        raw_out = output['data']['text/plain']
                        if isinstance(raw_out, list):
                            raw_out = '\n'.join(raw_out)
                        output_text = html.escape(str(raw_out)[:300])
                        display += f'            <pre>{output_text}</pre>\n'
                display += '        </div>\n'
            
            display += """    </div>
"""
        
        display += """
</div>
"""
        
        return display
    
    def ask_question(self, query: str, conversation_display: List) -> Tuple[List, str]:
        """Answer a question about the notebook with conversation history."""
        if not self.answering_system:
            error_msg = "❌ No document loaded. Please upload a document first."
            formatted_display = self._ensure_message_format(conversation_display)
            formatted_display.append({"role": "user", "content": query})
            formatted_display.append({"role": "assistant", "content": error_msg})
            return formatted_display, ""
        
        if not query or query.strip() == "":
            return conversation_display, ""
        
        try:
            # Convert incoming display to role/content format
            formatted_display = self._ensure_message_format(conversation_display)
            
            # Sync internal conversation history with display
            self.conversation_history = []
            for msg in formatted_display:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    self.conversation_history.append(msg)

            # Add the new user message to internal history
            self.conversation_history.append({"role": "user", "content": query})

            # Check if this is a casual greeting/small talk (no document context needed)
            is_casual = self._is_casual_conversation(query)
            
            if is_casual and self.groq_client:
                # Use Groq for natural conversation without document analysis
                try:
                    answer_text = self.groq_client.reason(
                        query=query,
                        context="User is having a casual conversation.",
                        conversation_history=self.conversation_history
                    )
                except Exception:
                    answer_text = self._get_fallback_greeting(query)
            elif is_casual:
                # Fallback friendly response without Groq
                answer_text = self._get_fallback_greeting(query)
            else:
                # Document-based Q&A
                response = self.answering_system.answer_question(
                    query,
                    top_k=8,
                    conversation_history=self.conversation_history
                )

                # Format answer
                answer_text = response.answer

                # Add citations if available
                if response.citations:
                    answer_text += "\n\n**📚 References:**\n"
                    for i, citation in enumerate(response.citations, 1):
                        answer_text += f"\n{i}. `{citation.cell_id}` [{citation.cell_type}]"
                        if citation.intent:
                            answer_text += f" - *{citation.intent}*"

                # Add confidence
                answer_text += f"\n\n*Confidence: {response.confidence:.0%}*"
                if response.has_hallucination_risk:
                    answer_text += " ⚠️ *Verify information*"

            # Add to both conversation history and display
            self.conversation_history.append({"role": "assistant", "content": answer_text})
            formatted_display.append({"role": "user", "content": query})
            formatted_display.append({"role": "assistant", "content": answer_text})

            return formatted_display, ""

        except Exception as e:
            formatted_display = self._ensure_message_format(conversation_display)
            formatted_display.append({"role": "user", "content": query})
            formatted_display.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
            return formatted_display, ""
    
    def _is_casual_conversation(self, query: str) -> bool:
        """Detect if query is casual conversation (greeting, small talk) vs document Q&A."""
        query_lower = query.lower().strip()
        
        # Greetings
        greetings = ['hi', 'hello', 'hey', 'howdy', 'greetings', 'good morning', 'good afternoon', 'good evening']
        if any(query_lower.startswith(g) for g in greetings):
            return True
        
        # Small talk / general questions
        small_talk = [
            "how are you", "how are u", "how's it going", "what's up", "sup",
            "how do i use", "how do i get started", "what can you do", "what are you",
            "who are you", "tell me about yourself", "introduce yourself",
            "thanks", "thank you", "great", "awesome", "nice", "cool",
            "lol", "haha", "ha ha"
        ]
        if any(small_talk_phrase in query_lower for small_talk_phrase in small_talk):
            return True
        
        # Questions that don't reference the document
        if query.startswith("?") or query.endswith("?"):
            if len(query.split()) < 4:  # Short questions likely casual
                return True
        
        return False
    
    def _get_fallback_greeting(self, query: str) -> str:
        """Generate a friendly fallback response for casual conversation."""
        query_lower = query.lower().strip()
        
        if any(q in query_lower for q in ['hi', 'hello', 'hey', 'greetings']):
            return "👋 Hey there! I'm ready to analyze your documents. Upload a notebook or Excel file to get started, and I can answer questions, generate summaries, and provide insights!"
        elif any(q in query_lower for q in ['how are you', "how's it going", "what's up"]):
            return "😊 I'm doing great, thanks for asking! Ready to dive into your documents. What would you like to know?"
        elif any(q in query_lower for q in ['what can you do', 'who are you', 'tell me about']):
            return "🤖 I'm an AI assistant specialized in analyzing Jupyter notebooks and Excel files. I can:\n- Summarize key findings\n- Answer questions about your data\n- Generate insights and keypoints\n- Provide data profiles and statistics\n\nUpload a file to get started!"
        elif any(q in query_lower for q in ['thanks', 'thank you', 'great', 'awesome']):
            return "😄 You're welcome! Happy to help. What else would you like to know about your document?"
        else:
            return "👋 I'm here to help! Upload a document and ask me anything about it. What would you like to explore?"
    
    def _ensure_message_format(self, conversation_display: List) -> List[Dict]:
        """Convert conversation display to Gradio ChatMessage format (role/content dicts)."""
        if not conversation_display:
            return []
        
        result = []
        for item in conversation_display:
            # Already in dict format
            if isinstance(item, dict) and "role" in item and "content" in item:
                result.append(item)
            # Old format: [user_text, assistant_text] tuple/list
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                result.append({"role": "user", "content": str(item[0])})
                result.append({"role": "assistant", "content": str(item[1])})
        
        return result
    
    # ==================== KILLER FEATURES ====================
    
    def generate_data_profile(self) -> str:
        """Generate comprehensive data profiling and statistics."""
        if not self.current_thread:
            return "❌ No document loaded."
        
        profile = """
<style>
.profile-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 8px;
    margin: 12px 0;
}
.metric {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    padding: 12px 16px;
    border-radius: 6px;
    margin: 6px;
    font-weight: 500;
}
.code-quality {
    background: #f0f9ff;
    border-left: 4px solid #0284c7;
    padding: 16px;
    margin: 12px 0;
    border-radius: 6px;
}
.insight-box {
    background: #fef3c7;
    border-left: 4px solid #f59e0b;
    padding: 16px;
    margin: 12px 0;
    border-radius: 6px;
}
</style>

<div class="profile-card">
    <h2>📊 Document Profile & Analytics</h2>
    <p>Comprehensive analysis of your notebook</p>
</div>
"""
        
        # Calculate metrics
        total_cells = len(self.current_thread.units)
        code_cells = sum(1 for u in self.current_thread.units if u.cell.cell_type == CellType.CODE)
        markdown_cells = total_cells - code_cells
        cells_with_output = sum(1 for u in self.current_thread.units if u.cell.outputs)
        cells_with_intent = sum(1 for u in self.current_thread.units if u.intent and u.intent != "[Pending intent inference]")
        
        total_lines = sum(len(u.cell.source.split('\n')) for u in self.current_thread.units)
        avg_cell_size = total_lines // max(code_cells, 1)
        
        profile += f"""
<div class="code-quality">
    <h3>📈 Key Metrics</h3>
    <div>
        <div class="metric">Total Cells: <strong>{total_cells}</strong></div>
        <div class="metric">Code Cells: <strong>{code_cells}</strong></div>
        <div class="metric">Documentation: <strong>{markdown_cells}</strong></div>
        <div class="metric">Cells with Output: <strong>{cells_with_output}</strong></div>
        <div class="metric">Total Lines: <strong>{total_lines}</strong></div>
        <div class="metric">Avg Cell Size: <strong>{avg_cell_size} lines</strong></div>
    </div>
</div>

<div class="insight-box">
    <h3>💡 Code Quality Insights</h3>
"""
        
        # Quality analysis
        insights = []
        
        if cells_with_output / max(code_cells, 1) > 0.8:
            insights.append("✅ <strong>Excellent output coverage:</strong> Most cells produce outputs")
        if cells_with_intent / total_cells > 0.7:
            insights.append("✅ <strong>Well-structured workflow:</strong> Clear intent in most cells")
        if code_cells < markdown_cells:
            insights.append("✅ <strong>Well documented:</strong> Good documentation-to-code ratio")
        if total_lines > 500:
            insights.append("⚠️ <strong>Large notebook:</strong> Consider breaking into smaller modules")
        if avg_cell_size > 30:
            insights.append("⚠️ <strong>Large cells:</strong> Some cells could be smaller for clarity")
        
        if not insights:
            insights.append("ℹ️ Standard notebook structure detected")
        
        for insight in insights:
            profile += f"<p>{insight}</p>\n"
        
        profile += """
</div>

<div class="insight-box">
    <h3>🔍 Intent Distribution</h3>
"""
        
        intent_counts = {}
        for unit in self.current_thread.units:
            if unit.intent and unit.intent != "[Pending intent inference]":
                intent = unit.intent.split()[0]  # Get first word of intent
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            profile += f"<p>• <strong>{intent}:</strong> {count} cells</p>\n"
        
        profile += """
</div>

<div class="insight-box">
    <h3>📦 Dependencies & Imports</h3>
"""
        
        imports = set()
        for unit in self.current_thread.units:
            if unit.cell.cell_type == CellType.CODE:
                source = unit.cell.source if isinstance(unit.cell.source, str) else ''.join(unit.cell.source)
                if 'import ' in source:
                    for line in source.split('\n'):
                        if line.strip().startswith(('import ', 'from ')):
                            # Extract module name
                            module = line.split('import')[0].replace('from', '').strip()
                            if module:
                                imports.add(module)
        
        if imports:
            for imp in sorted(imports)[:10]:
                profile += f"<p>• <code>{imp}</code></p>\n"
        else:
            profile += "<p>No imports detected</p>\n"
        
        profile += """
</div>
"""
        
        return profile
    
    def export_analysis(self) -> str:
        """Export analysis results."""
        if not self.current_thread:
            return "❌ No document loaded."
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{self.current_file_name or 'notebook'}_{timestamp}.md"
        
        # Create markdown report
        report = f"""# Document Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
{self.keypoints_cache or "Key insights would be generated here."}

## Key Metrics
- Total Cells: {len(self.current_thread.units)}
- Code Cells: {sum(1 for u in self.current_thread.units if u.cell.cell_type == CellType.CODE)}
- Documentation Cells: {sum(1 for u in self.current_thread.units if u.cell.cell_type == CellType.MARKDOWN)}

## Questions Asked
"""
        
        for msg in self.conversation_history:
            if msg["role"] == "user":
                report += f"\n- {msg['content'][:100]}"
        
        # Save to file
        with open(filename, 'w') as f:
            f.write(report)
        
        return f"✅ Report exported to `{filename}`"
    
    def advanced_search(self, search_term: str) -> str:
        """Advanced search across all cells."""
        if not self.current_thread or not search_term:
            return "❌ No document loaded or search term empty."
        
        results = []
        search_lower = search_term.lower()
        
        for i, unit in enumerate(self.current_thread.units, 1):
            source_text = unit.cell.source if isinstance(unit.cell.source, str) else ''.join(unit.cell.source)
            if search_lower in source_text.lower():
                results.append({
                    "cell": i,
                    "type": unit.cell.cell_type,
                    "intent": unit.intent,
                    "snippet": source_text[:150]
                })
        
        if not results:
            return f"No results found for '{search_term}'"
        
        output = f"<h3>🔍 Found {len(results)} matches for '{search_term}'</h3>\n"
        
        for r in results[:10]:
            output += f"""
<div style="background: #f0f4f8; padding: 12px; margin: 8px 0; border-radius: 6px; border-left: 4px solid #0284c7;">
<strong>Cell {r['cell']}</strong> [{r['type'].upper()}] {r['intent']}<br/>
<code style="font-size: 0.85em;">{r['snippet']}...</code>
</div>
"""
        
        return output
    
    def get_recommendations(self) -> str:
        """Generate smart recommendations."""
        if not self.current_thread:
            return "❌ No document loaded."
        
        recommendations = """
<style>
.rec-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 20px;
    border-radius: 8px;
    margin: 12px 0;
}
.rec-item {
    background: rgba(0,0,0,0.2);
    padding: 12px;
    margin: 8px 0;
    border-radius: 6px;
}
</style>

<div class="rec-card">
    <h2>⭐ AI-Powered Recommendations</h2>
</div>
"""
        
        recs = []
        
        code_cells = sum(1 for u in self.current_thread.units if u.cell.cell_type == CellType.CODE)
        markdown_cells = sum(1 for u in self.current_thread.units if u.cell.cell_type == CellType.MARKDOWN)
        
        if code_cells > 20:
            recs.append("🔄 Consider modularizing code into separate files/functions")
        if markdown_cells == 0:
            recs.append("📝 Add documentation cells for better clarity")
        if len(self.current_thread.units) > 50:
            recs.append("📚 This notebook is large - consider splitting into multiple notebooks")
        
        # Check for common issues
        large_cells = sum(1 for u in self.current_thread.units if len(u.cell.source) > 1000)
        if large_cells > 0:
            recs.append(f"✂️ {large_cells} cells are very large - consider breaking them down")
        
        cells_without_output = sum(1 for u in self.current_thread.units if u.cell.cell_type == CellType.CODE and not u.cell.outputs)
        if cells_without_output > code_cells * 0.3:
            recs.append("⚠️ Many code cells don't have outputs - ensure cells are executable")
        
        if not recs:
            recs.append("✅ Notebook follows best practices!")
        
        for i, rec in enumerate(recs, 1):
            recommendations += f'<div class="rec-item">{i}. {rec}</div>\n'
        
        return recommendations
    
    def _excel_to_cells(self, excel_path: str) -> List[Cell]:
        """Convert Excel file to notebook-like cells."""
        from src.models import Cell, CellType
        
        cells = []
        xl = pd.ExcelFile(excel_path)
        
        # Add overview cell
        cells.append(Cell(
            cell_id="excel_overview",
            cell_type=CellType.MARKDOWN,
            source=f"# Excel Document Analysis\n\nSheets: {', '.join(xl.sheet_names)}\nTotal Sheets: {len(xl.sheet_names)}",
            outputs=[]
        ))
        
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            
            # Sheet header
            cells.append(Cell(
                cell_id=f"sheet_{sheet_name}_header",
                cell_type=CellType.MARKDOWN,
                source=f"## Sheet: {sheet_name}\n\n**Dimensions:** {df.shape[0]} rows × {df.shape[1]} columns",
                outputs=[]
            ))
            
            # Column info
            col_info = "\n".join([f"- {col}: {dtype}" for col, dtype in df.dtypes.items()])
            cells.append(Cell(
                cell_id=f"sheet_{sheet_name}_columns",
                cell_type=CellType.MARKDOWN,
                source=f"### Columns\n{col_info}",
                outputs=[]
            ))
            
            # Data preview
            cells.append(Cell(
                cell_id=f"data_{sheet_name}_preview",
                cell_type=CellType.CODE,
                source=f"# Preview of {sheet_name}\ndf_{sheet_name}.head(10)",
                outputs=[{"data": {"text/plain": df.head(10).to_string()}}]
            ))
            
            # Statistics
            if df.select_dtypes(include=['number']).shape[1] > 0:
                stats = df.describe().to_string()
                cells.append(Cell(
                    cell_id=f"stats_{sheet_name}",
                    cell_type=CellType.CODE,
                    source=f"# Statistics for {sheet_name}\ndf_{sheet_name}.describe()",
                    outputs=[{"data": {"text/plain": stats}}]
                ))
        
        return cells
    
    def get_excel_display(self, excel_path: str) -> str:
        """Get Microsoft Excel-like styled spreadsheet content."""
        xl = pd.ExcelFile(excel_path)
        sheet_names = xl.sheet_names
        
        if not sheet_names:
            return "No sheets found in Excel file."
        
        primary_sheet = sheet_names[0]
        df = xl.parse(primary_sheet)
        
        display = """
<style>
.excel-container {
    font-family: 'Calibri', 'Arial', sans-serif;
    padding: 16px;
    background: white;
}

.excel-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
    padding: 12px 16px;
    background: linear-gradient(135deg, #2d7f38 0%, #4caf50 100%);
    border-radius: 4px;
    color: white;
}

.excel-header h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 500;
}

.excel-header-subtitle {
    color: rgba(255,255,255,0.95);
    font-size: 12px;
    margin-top: 2px;
}

.excel-toolbar {
    display: flex;
    gap: 8px;
    padding: 12px 0;
    border-bottom: 1px solid #e0e0e0;
    margin-bottom: 16px;
    overflow-x: auto;
}

.excel-tab {
    padding: 8px 16px;
    background: white;
    border: 1px solid #d0d0d0;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    cursor: pointer;
    font-weight: 500;
    color: #666;
    font-size: 13px;
    white-space: nowrap;
}

.excel-tab.active {
    background: white;
    color: #2d7f38;
    border-color: #2d7f38;
    border-bottom: 2px solid white;
    margin-bottom: -1px;
}

.excel-grid-wrapper {
    overflow-x: auto;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    background: white;
}

.excel-grid table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.excel-grid th {
    background: #f3f3f3;
    border: 1px solid #d0d0d0;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    color: #333;
    position: sticky;
    top: 0;
    z-index: 10;
    min-width: 80px;
}

.excel-grid td {
    border: 1px solid #e0e0e0;
    padding: 8px 12px;
    color: #333;
    background: white;
}

.excel-grid tr:nth-child(even) td {
    background: #f9f9f9;
}

.excel-grid tr:hover td {
    background: #e8f5e9;
}

.excel-row-header {
    background: #f3f3f3;
    border: 1px solid #d0d0d0;
    padding: 8px 12px;
    font-weight: 600;
    color: #666;
    text-align: center;
    width: 40px;
    min-width: 40px;
}

.excel-stats {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}

.excel-stat {
    flex: 1;
    min-width: 120px;
    background: #f9f9f9;
    border: 1px solid #d0d0d0;
    padding: 12px;
    border-radius: 4px;
    text-align: center;
}

.excel-stat-value {
    font-size: 20px;
    font-weight: 600;
    color: #2d7f38;
}

.excel-stat-label {
    font-size: 12px;
    color: #666;
    margin-top: 6px;
}

.excel-data-info {
    background: #f0f7f0;
    border-left: 4px solid #2d7f38;
    padding: 12px;
    margin-bottom: 16px;
    border-radius: 4px;
    font-size: 13px;
}

.excel-data-info strong {
    color: #2d7f38;
}
</style>

<div class="excel-container">
    <div class="excel-header">
        <div>
            <h1>📊 Excel Data Viewer</h1>
            <div class="excel-header-subtitle">Microsoft Excel-style Professional Spreadsheet</div>
        </div>
    </div>
"""
        
        display += f"""
    <div class="excel-stats">
        <div class="excel-stat">
            <div class="excel-stat-value">{len(df)}</div>
            <div class="excel-stat-label">Rows</div>
        </div>
        <div class="excel-stat">
            <div class="excel-stat-value">{len(df.columns)}</div>
            <div class="excel-stat-label">Columns</div>
        </div>
        <div class="excel-stat">
            <div class="excel-stat-value">{df.memory_usage(deep=True).sum() / 1024:.1f} KB</div>
            <div class="excel-stat-label">Size</div>
        </div>
        <div class="excel-stat">
            <div class="excel-stat-value">{df.isnull().sum().sum()}</div>
            <div class="excel-stat-label">Missing</div>
        </div>
    </div>

    <div class="excel-data-info">
        <strong>📋 Data Summary:</strong> {len(df)} rows × {len(df.columns)} columns | Dtypes: {', '.join(map(str, df.dtypes.unique()))}
    </div>

    <div class="excel-toolbar">
        <div class="excel-tab active">{primary_sheet}</div>
"""
        
        for sheet in sheet_names[1:]:
            display += f'        <div class="excel-tab">{sheet}</div>\n'
        
        display += """    </div>

    <div class="excel-grid-wrapper">
        <table class="excel-grid">
            <thead>
                <tr>
                    <th class="excel-row-header"></th>
"""
        
        for col in df.columns:
            display += f"                    <th>{col}</th>\n"
        
        display += """                </tr>
            </thead>
            <tbody>
"""
        
        for idx, row in df.head(100).iterrows():
            display += f"                <tr>\n                    <td class='excel-row-header'>{idx + 1}</td>\n"
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    display += "                    <td style='color: #ccc;'>—</td>\n"
                else:
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:,.2f}" if isinstance(value, float) else str(value)
                    else:
                        formatted_value = str(value)[:50]
                    display += f"                    <td>{formatted_value}</td>\n"
            display += "                </tr>\n"
        
        if len(df) > 100:
            display += f"""                <tr>
                    <td colspan="{len(df.columns) + 1}" style="text-align: center; color: #999; padding: 12px;">
                        ... and {len(df) - 100} more rows
                    </td>
                </tr>
"""
        
        display += """            </tbody>
        </table>
    </div>

</div>
"""
        
        return display


def create_gradio_app():
    """Create and return the enhanced Gradio interface."""
    agent = NotebookAgentUI()

    # Auto-initialize Groq if key present in environment but client wasn't created earlier
    try:
        if not agent.groq_client:
            groq_key = os.getenv("GROQ_API_KEY")
            # Fallback: read .env directly if load_dotenv didn't pick it up
            if not groq_key:
                env_path = Path(__file__).parent.parent / '.env'
                if env_path.exists():
                    content = env_path.read_text(encoding='utf-8')
                    for line in content.splitlines():
                        line = line.strip()
                        if line.startswith('GROQ_API_KEY=') and not line.startswith('#'):
                            groq_key = line.split('=', 1)[1].strip()
                            if groq_key:
                                break

            if groq_key:
                try:
                    agent.set_groq_key(groq_key, True)
                except Exception:
                    pass
    except Exception:
        pass
    
    # Custom CSS for better styling
    custom_css = """
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-box {
        padding: 1rem;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .upload-section {
        text-align: center;
        padding: 2rem;
        border: 3px dashed #667eea;
        border-radius: 10px;
        background: #f8f9ff;
    }
    """
    
    with gr.Blocks(title="Context Thread Agent", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.HTML("""
        <div class="main-header">
            <h1>🧵 Context Thread Agent</h1>
            <p style="font-size: 1.2rem; margin-top: 1rem;">
                AI-Powered Document Analysis & Q&A System
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("""
                ## 🎯 What is Context Thread Agent?
                
                Context Thread Agent is an **intelligent document analysis platform** that helps you understand and extract insights from complex Jupyter notebooks and Excel spreadsheets. Using advanced AI (powered by **Groq LLM**), it provides:
                
                ### 🚀 Major Use Cases:
                
                - **📊 Data Analysis Review**: Understand complex analytical workflows instantly
                - **🔍 Code Audit**: Verify assumptions and logic in data science notebooks
                - **📈 Excel Report Analysis**: Extract insights from large spreadsheets
                - **🤖 Automated Documentation**: Generate summaries and key findings
                - **💡 Knowledge Extraction**: Ask questions about methodology and results
                - **🔗 Dependency Tracking**: Understand how different parts connect
                - **✅ Quality Assurance**: Validate calculations and transformations
                
                ### ✨ Key Features:
                - ✓ **100% Grounded Answers** - No hallucinations, only facts from your document
                - ✓ **Citation-Based** - Every answer references specific cells
                - ✓ **Context-Aware** - Understands relationships between code sections
                - ✓ **Conversation Memory** - Maintains context across questions
                - ✓ **Key Insights Generation** - AI-powered summary of main points
                - ✓ **Fast & Free** - Powered by Groq's lightning-fast inference
                """)
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="upload-section">
                    <h3>📤 Quick Start</h3>
                    <p>Upload your document and start exploring</p>
                </div>
                """)
                
                file_input = gr.File(
                    label="Upload Your Document",
                    file_types=[".ipynb", ".xlsx", ".xls"],
                    type="filepath",
                    elem_classes="upload-input"
                )
                upload_btn = gr.Button(
                    "📤 Upload & Analyze", 
                    variant="primary", 
                    size="lg",
                    scale=2
                )
                
                upload_status = gr.Markdown("### 📋 Status\n\nReady to upload...")
                
                # Groq status - show only status if enabled, otherwise show input
                if agent.groq_client:
                    groq_status = gr.Markdown("### 🚀 Groq Configuration\n\n✅ **Groq is enabled and ready!**\n\nYour Groq API key has been loaded from environment. Advanced reasoning will be used for analysis.")
                    # Hidden inputs for compatibility
                    groq_key_input = gr.Textbox(visible=False)
                    groq_toggle = gr.Checkbox(visible=False)
                    set_groq_btn = gr.Button(visible=False)
                else:
                    # Show input if Groq not enabled
                    groq_key_input = gr.Textbox(
                        label="Groq API Key",
                        placeholder="Paste your Groq key (gsk_...)",
                        type="password"
                    )
                    groq_toggle = gr.Checkbox(label="Use Groq for reasoning", value=False)
                    set_groq_btn = gr.Button("Set Groq Key", variant="secondary")
                    groq_status = gr.Markdown("⚠️ **Groq not configured.** Add your key and click 'Set Groq Key' to enable advanced reasoning.")
                    
                    # Wire the set key button only if inputs are visible
                    set_groq_btn.click(agent.set_groq_key, inputs=[groq_key_input, groq_toggle], outputs=[groq_status])
        
        gr.Markdown("---")
        
        # Main interface (hidden until upload)
        with gr.Column(visible=False) as main_interface:
            gr.Markdown("## 💼 Analysis Workspace")
            
            with gr.Row():
                # Left side: Document viewer
                with gr.Column(scale=1):
                    gr.Markdown("### 📓 Document Viewer")
                    
                    with gr.Tabs():
                        with gr.Tab("📄 Content"):
                            notebook_display = gr.HTML(
                                value="",
                                label="Document Content",
                                elem_classes="notebook-viewer"
                            )
                        
                        with gr.Tab("🔑 Key Points"):
                            keypoints_btn = gr.Button(
                                "🔄 Generate Key Insights", 
                                variant="secondary",
                                size="lg"
                            )
                            gr.Markdown("*This may take 10-30 seconds for comprehensive analysis...*")
                            keypoints_display = gr.Markdown(
                                value="",
                                label="Key Insights"
                            )
                        
                        with gr.Tab("📊 Analytics"):
                            analytics_btn = gr.Button("📊 Generate Profile", variant="secondary", size="lg")
                            analytics_display = gr.Markdown(value="", label="Analytics")
                        
                        with gr.Tab("⭐ Recommendations"):
                            rec_btn = gr.Button("💡 Get Recommendations", variant="secondary", size="lg")
                            rec_display = gr.Markdown(value="", label="Recommendations")
                        
                        with gr.Tab("🔍 Advanced Search"):
                            search_input = gr.Textbox(
                                label="Search Term",
                                placeholder="Search in all cells...",
                                lines=1
                            )
                            search_btn = gr.Button("🔎 Search", variant="secondary")
                            search_display = gr.Markdown(value="", label="Search Results")
                        
                        with gr.Tab("📥 Export"):
                            export_btn = gr.Button("📥 Export Analysis Report", variant="secondary", size="lg")
                            export_display = gr.Markdown(value="", label="Export Status")
                
                # Right side: Q&A Interface
                with gr.Column(scale=1):
                    gr.Markdown("### 💬 Ask Questions")
                    
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        elem_classes="chat-box"
                    )
                    
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., 'What are the main findings?' or 'Why was Q4 data removed?'",
                            lines=2,
                            scale=4
                        )
                        ask_btn = gr.Button("🤖 Ask", variant="primary", scale=1)
                    
                    gr.Markdown("""
                    **💡 Example Questions:**
                    - What is this document about?
                    - What are the key findings?
                    - Why was [specific data] removed?
                    - How was [metric] calculated?
                    - What patterns were found?
                    - Are there any data quality issues?
                    """)
        
        # Event handlers
        def on_upload(file):
            status, show_interface, notebook_content, keypoints = agent.load_notebook(file)
            return (
                status,
                gr.update(visible=show_interface),
                notebook_content,
                keypoints
            )
        
        upload_btn.click(
            fn=on_upload,
            inputs=[file_input],
            outputs=[upload_status, main_interface, notebook_display, keypoints_display]
        )
        
        # Keypoints generation with loading state
        def generate_with_loading():
            return "⏳ **Analyzing document and generating insights...**\n\nThis may take 10-30 seconds depending on document complexity."
        
        keypoints_btn.click(
            fn=generate_with_loading,
            inputs=[],
            outputs=[keypoints_display]
        ).then(
            fn=agent.generate_keypoints,
            inputs=[],
            outputs=[keypoints_display]
        )
        
        # Analytics tab
        analytics_btn.click(
            fn=agent.generate_data_profile,
            inputs=[],
            outputs=[analytics_display]
        )
        
        # Recommendations tab
        rec_btn.click(
            fn=agent.get_recommendations,
            inputs=[],
            outputs=[rec_display]
        )
        
        # Advanced search
        search_btn.click(
            fn=agent.advanced_search,
            inputs=[search_input],
            outputs=[search_display]
        )
        
        # Export
        export_btn.click(
            fn=agent.export_analysis,
            inputs=[],
            outputs=[export_display]
        )
        
        # Q&A interaction
        ask_btn.click(
            fn=agent.ask_question,
            inputs=[query_input, chatbot],
            outputs=[chatbot, query_input]
        )
        
        query_input.submit(
            fn=agent.ask_question,
            inputs=[query_input, chatbot],
            outputs=[chatbot, query_input]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
