"""
Gradio UI for Context Thread Agent - Enhanced Version
Professional interface for uploading notebooks/Excel and asking questions
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
    """Enhanced Gradio UI for the Context Thread Agent."""
    
    def __init__(self):
        self.current_thread = None
        self.current_indexer = None
        self.current_engine = None
        self.answering_system = None
        self.conversation_history = []
        self.groq_client = None
        self.keypoints_generated = False
        self.keypoints_cache = None
        
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
            
            # Cleanup
            Path(temp_path).unlink()
            
            # Get notebook preview
            notebook_preview = self.get_notebook_display()
            
            status_msg = f"""
### ✅ File Loaded Successfully!

**Document Statistics:**
- Total sections: {len(cells)}
- Code sections: {sum(1 for c in cells if c.cell_type == 'code')}
- Documentation: {sum(1 for c in cells if c.cell_type == 'markdown')}
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
                all_context.append(unit.cell.source[:500])
                if unit.cell.outputs:
                    for output in unit.cell.outputs[:1]:
                        if 'text' in output:
                            all_context.append(f"Output: {output['text'][:200]}")
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
    
    def get_notebook_display(self) -> str:
        """Get formatted notebook content for display."""
        if not self.current_thread:
            return "No document loaded."
        
        display = "# 📄 Document Content\n\n"
        for i, unit in enumerate(self.current_thread.units, 1):
            display += f"### Cell {i}: `{unit.cell.cell_id}` [{unit.cell.cell_type}]\n"
            if unit.intent and unit.intent != "[Pending intent inference]":
                display += f"**Intent:** *{unit.intent}*\n\n"
            
            if unit.cell.cell_type == 'code':
                display += f"```python\n{unit.cell.source}\n```\n\n"
            else:
                display += f"{unit.cell.source}\n\n"
            
            if unit.cell.outputs:
                display += "<details><summary>📊 Output</summary>\n\n"
                for output in unit.cell.outputs[:2]:
                    if 'text' in output:
                        display += f"```\n{output['text'][:500]}\n```\n"
                    elif 'data' in output and 'text/plain' in output['data']:
                        display += f"```\n{str(output['data']['text/plain'])[:500]}\n```\n"
                display += "</details>\n\n"
            
            display += "---\n\n"
        
        return display
    
    def ask_question(self, query: str, conversation_display: List) -> Tuple[List, str]:
        """Answer a question about the notebook with conversation history."""
        if not self.answering_system:
            return conversation_display + [[query, "❌ No document loaded. Please upload a document first."]], ""
        
        if not query or query.strip() == "":
            return conversation_display, ""
        
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": query})
            
            # Get answer with conversation context
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
            
            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": response.answer})
            
            # Update conversation display
            conversation_display = conversation_display + [[query, answer_text]]
            
            return conversation_display, ""
        
        except Exception as e:
            return conversation_display + [[query, f"❌ Error: {str(e)}"]], ""
    
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


def create_gradio_app():
    """Create and return the enhanced Gradio interface."""
    agent = NotebookAgentUI()
    
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
                            notebook_display = gr.Markdown(
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
