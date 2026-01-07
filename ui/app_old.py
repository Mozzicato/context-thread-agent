"""
Gradio UI for Context Thread Agent.
Interactive interface for uploading notebooks and asking questions.
"""

import gradio as gr
import json
import tempfile
from pathlib import Path
from typing import Tuple, List, Dict
from src.models import Cell

from src.parser import NotebookParser
from src.dependencies import ContextThreadBuilder
from src.indexing import FAISSIndexer
from src.retrieval import RetrievalEngine, ContextBuilder
from src.reasoning import ContextualAnsweringSystem
from src.intent import ContextThreadEnricher
from src.groq_integration import GroqReasoningEngine
import pandas as pd


class NotebookAgentUI:
    """Gradio UI for the Context Thread Agent."""
    
    def __init__(self):
        self.current_thread = None
        self.current_indexer = None
        self.current_engine = None
        self.answering_system = None
        self.conversation_history = []  # To maintain context across questions
    
    def load_notebook(self, notebook_file) -> str:
        """Load and index a notebook or Excel file."""
        try:
            if notebook_file is None:
                return "❌ No file provided"
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix=Path(notebook_file).suffix if isinstance(notebook_file, str) else ".ipynb", delete=False) as f:
                if isinstance(notebook_file, str):
                    # File path
                    f.write(open(notebook_file, 'rb').read())
                else:
                    # Uploaded file
                    f.write(notebook_file.read())
                temp_path = f.name
            
            file_ext = Path(temp_path).suffix.lower()
            
            if file_ext == '.ipynb':
                # Parse notebook
                parser = NotebookParser()
                result = parser.parse_file(temp_path)
                cells = result['cells']
            elif file_ext in ['.xlsx', '.xls']:
                # Convert Excel to pseudo-cells
                cells = self._excel_to_cells(temp_path)
            else:
                return "❌ Unsupported file type. Please upload .ipynb or .xlsx/.xls"
            
            # Build context thread
            builder = ContextThreadBuilder(
                notebook_name=Path(temp_path).stem,
                thread_id=f"thread_{id(self)}"
            )
            builder.add_cells(cells)
            self.current_thread = builder.build()
            
            # Enrich with intents (heuristic, not LLM for speed)
            enricher = ContextThreadEnricher(infer_intents=True)
            self.current_thread = enricher.enrich(self.current_thread)
            
            # Index
            self.current_indexer = FAISSIndexer()
            self.current_indexer.add_multiple(self.current_thread.units)
            
            # Setup retrieval and reasoning
            self.current_engine = RetrievalEngine(self.current_thread, self.current_indexer)
            self.answering_system = ContextualAnsweringSystem(self.current_engine)
            
            # Cleanup
            Path(temp_path).unlink()
            
            return f"""
✅ **File Loaded Successfully!**

**Stats:**
- Total cells/sections: {len(cells)}
- Code cells: {sum(1 for c in cells if c.cell_type == 'code')}
- Markdown cells: {sum(1 for c in cells if c.cell_type == 'markdown')}
- Indexed: ✓

Ready to ask questions! 🎯
"""
        
        except Exception as e:
            return f"❌ Error loading file: {str(e)}"
    
    def generate_keypoints(self) -> str:
        """Generate key points summary of the notebook."""
        if not self.answering_system:
            return "❌ No notebook loaded."
        
        try:
            # Use the reasoning engine to summarize keypoints
            query = "Summarize the key points and main insights from this notebook."
            response = self.answering_system.answer_question(query, top_k=10)  # More context for summary
            
            keypoints = f"**Key Points Summary:**\n\n{response.answer}"
            return keypoints
        except Exception as e:
            return f"❌ Error generating keypoints: {str(e)}"
    
    def get_notebook_display(self) -> str:
        """Get formatted notebook content for display."""
        if not self.current_thread:
            return "No notebook loaded."
        
        display = ""
        for i, unit in enumerate(self.current_thread.units, 1):
            display += f"### Cell {i}: {unit.cell.cell_id} [{unit.cell.cell_type}]\n"
            if unit.intent and unit.intent != "[Pending intent inference]":
                display += f"**Intent:** {unit.intent}\n\n"
            display += f"```\n{unit.cell.source}\n```\n\n"
            if unit.cell.outputs:
                display += "**Output:**\n"
                for output in unit.cell.outputs:
                    if 'text' in output:
                        display += f"```\n{output['text']}\n```\n"
                    elif 'data' in output and 'text/plain' in output['data']:
                        display += f"```\n{output['data']['text/plain']}\n```\n"
                display += "\n"
        
        return display
    
    def ask_question(self, query: str) -> Tuple[str, str, str]:
        """Answer a question about the notebook."""
        if not self.answering_system:
            return (
                "❌ No notebook loaded. Please upload a notebook first.",
                "",
                ""
            )
        
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": query})
            
            # Get answer
            response = self.answering_system.answer_question(query, top_k=5)
            
            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": response.answer})
            
            # Format answer
            answer_text = f"**Answer:**\n\n{response.answer}"
            
            # Format citations
            if response.citations:
                citations_text = "**Citations:**\n\n"
                for i, citation in enumerate(response.citations, 1):
                    citations_text += f"{i}. **{citation.cell_id}** [{citation.cell_type}]\n"
                    if citation.intent:
                        citations_text += f"   *Intent: {citation.intent}*\n"
                    citations_text += f"   ```\n{citation.content_snippet}\n   ```\n\n"
            else:
                citations_text = "*No specific cells cited*"
            
            # Format context
            context_text = "**Retrieved Context:**\n\n"
            for unit in response.retrieved_units:
                context_text += f"### {unit.cell.cell_id} [{unit.cell.cell_type}]\n"
                if unit.intent != "[Pending intent inference]":
                    context_text += f"**Intent:** {unit.intent}\n\n"
                if unit.dependencies:
                    context_text += f"**Depends on:** {', '.join(unit.dependencies)}\n\n"
                context_text += f"```python\n{unit.cell.source[:300]}\n```\n\n"
            
            context_text += f"\n**Confidence:** {response.confidence:.2%}\n"
            context_text += f"**Hallucination Risk:** {'⚠️ Yes' if response.has_hallucination_risk else '✅ No'}"
            
            return (answer_text, citations_text, context_text)
        
        except Exception as e:
            return (f"❌ Error: {str(e)}", "", "")
    
    def _excel_to_cells(self, excel_path: str) -> List[Cell]:
        """Convert Excel file to notebook-like cells."""
        from src.models import Cell, CellType
        
        cells = []
        
        # Read all sheets
        xl = pd.ExcelFile(excel_path)
        
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            
            # Create a markdown cell for sheet name
            cells.append(Cell(
                cell_id=f"sheet_{sheet_name}",
                cell_type=CellType.MARKDOWN,
                source=f"# Sheet: {sheet_name}",
                outputs=[]
            ))
            
            # Create a code cell for data loading
            cells.append(Cell(
                cell_id=f"data_{sheet_name}",
                cell_type=CellType.CODE,
                source=f"# Data from {sheet_name}\ndf_{sheet_name} = pd.read_excel('{excel_path}', sheet_name='{sheet_name}')",
                outputs=[{"data": {"text/plain": str(df.head())}}]
            ))
            
            # Create cells for basic analysis
            cells.append(Cell(
                cell_id=f"shape_{sheet_name}",
                cell_type=CellType.CODE,
                source=f"print(f'Shape: {df.shape}')",
                outputs=[{"text": f"Shape: {df.shape}"}]
            ))
            
            cells.append(Cell(
                cell_id=f"info_{sheet_name}",
                cell_type=CellType.CODE,
                source=f"df_{sheet_name}.info()",
                outputs=[{"text": str(df.dtypes)}]
            ))
        
        return cells


def create_gradio_app():
    """Create and return the Gradio interface."""
    agent = NotebookAgentUI()
    
    with gr.Blocks(title="Context Thread Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧵 Context Thread Agent")
        gr.Markdown("""
**An AI-powered notebook copilot for analytical workflows**

Upload your Jupyter notebook (.ipynb) or Excel file (.xlsx/.xls) and ask questions about your data analysis. 
The agent understands your notebook's context, dependencies, and reasoning—providing grounded answers with citations.

### Key Features:
- ✅ **Context-Aware**: Answers based only on your notebook content
- ✅ **Citation-Based**: Every claim references specific cells
- ✅ **Dependency-Aware**: Understands how cells relate
- ✅ **No Hallucinations**: Grounded in your actual analysis
- ✅ **Fast & Free**: Powered by Groq AI

### Major Uses:
- **Audit Analysis**: Verify assumptions and decisions in complex notebooks
- **Code Review**: Understand data transformations and logic flows
- **Documentation**: Generate insights summaries with evidence
- **Debugging**: Trace errors through dependent cells
- **Collaboration**: Share verifiable insights from your work
""")
        
        # Upload section
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload Notebook or Excel",
                    file_types=[".ipynb", ".xlsx", ".xls"],
                    type="filepath"
                )
                upload_btn = gr.Button("📤 Upload & Analyze", variant="primary", size="lg")
            with gr.Column(scale=1):
                upload_status = gr.Markdown("### Status\n\nReady to upload...")
        
        # After upload, show the main interface
        with gr.Row(visible=False) as main_interface:
            # Left side: Notebook viewer
            with gr.Column(scale=1):
                gr.Markdown("### 📓 Notebook Viewer")
                notebook_display = gr.Markdown("")
                
                keypoints_btn = gr.Button("🔑 Generate Key Points", variant="secondary")
                keypoints_display = gr.Markdown("")
                
            # Right side: Question answering
            with gr.Column(scale=1):
                gr.Markdown("### ❓ Ask Questions")
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., 'Why did we remove Q4 data?' or 'What are the key findings?'",
                    lines=3
                )
                ask_btn = gr.Button("🤖 Get Answer", variant="primary")
                
                answer_display = gr.Markdown("")
                citations_display = gr.Markdown("")
                context_display = gr.Markdown("")
        
        # Event handlers
        def on_upload(file):
            status = agent.load_notebook(file)
            if "✅" in status:
                return status, gr.update(visible=True), agent.get_notebook_display(), ""
            else:
                return status, gr.update(visible=False), "", ""
        
        upload_btn.click(
            fn=on_upload,
            inputs=[file_input],
            outputs=[upload_status, main_interface, notebook_display, keypoints_display]
        )
        
        keypoints_btn.click(
            fn=lambda: agent.generate_keypoints(),
            inputs=[],
            outputs=[keypoints_display]
        ).then(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[keypoints_display]
        )
        
        ask_btn.click(
            fn=agent.ask_question,
            inputs=[query_input],
            outputs=[answer_display, citations_display, context_display]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
