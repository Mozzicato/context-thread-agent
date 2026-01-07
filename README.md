# 🧵 Context Thread Agent

**AI-Powered Document Analysis & Q&A System for Jupyter Notebooks and Excel Files**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![Groq](https://img.shields.io/badge/AI-Groq%20LLM-green.svg)](https://groq.com/)

---

## 🎯 What is Context Thread Agent?

Context Thread Agent is an **intelligent document analysis platform** that helps you understand and extract insights from complex Jupyter notebooks and Excel spreadsheets. Using advanced AI powered by **Groq's lightning-fast LLM**, it provides:

✅ **100% Grounded Answers** - No hallucinations, only facts from your document  
✅ **Citation-Based Responses** - Every answer references specific cells/sections  
✅ **Context-Aware Analysis** - Understands relationships between code sections  
✅ **Conversation Memory** - Maintains context across multiple questions  
✅ **Key Insights Generation** - AI-powered summary of main points  
✅ **Professional UI** - Split-screen viewer with intuitive Q&A interface  

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key (free at console.groq.com)
export GROQ_API_KEY="your_key_here"

# 3. Generate demo files
python generate_demo_files.py

# 4. Launch the application
python main.py ui --port 7860
```

Open your browser to **http://localhost:7860**

📖 **Detailed Setup:** See [QUICKSTART.md](QUICKSTART.md)

---

## 💼 Major Use Cases

| Use Case | Description |
|----------|-------------|
| 📊 **Data Analysis Review** | Understand complex analytical workflows instantly |
| 🔍 **Code Audit** | Verify assumptions and logic in data science notebooks |
| 📈 **Excel Report Analysis** | Extract insights from large spreadsheets |
| 🤖 **Automated Documentation** | Generate summaries and key findings |
| 💡 **Knowledge Extraction** | Ask questions about methodology and results |
| 🔗 **Dependency Tracking** | Understand how different code sections connect |
| ✅ **Quality Assurance** | Validate calculations and transformations |

---

## ✨ Key Features

### 1. Professional Homepage
- Clear platform introduction
- Comprehensive use case showcase
- Prominent upload section
- Feature highlights

### 2. Split-Screen Workspace
**Left Panel - Document Viewer:**
- 📄 Browse full document with syntax highlighting
- 🔑 Generate AI-powered key insights (10-30 seconds)

**Right Panel - Q&A Interface:**
- 💬 Chatbot-style conversation
- 📚 Automatic citations
- ✓ Confidence scores
- 🧠 Context-aware responses

### 3. Enhanced AI Capabilities
- **Groq Integration:** Lightning-fast inference (< 3 seconds)
- **Conversation History:** Maintains context across questions
- **Key Points Generator:** Comprehensive document summarization
- **Citation Extraction:** References specific cells automatically

### 4. Smart Document Processing
- **Jupyter Notebooks:** Full code, markdown, and output analysis
- **Excel Files:** Multi-sheet support with statistics
- **Intent Recognition:** Understands purpose of code sections
- **Dependency Tracking:** Maps relationships between cells

---

## 📁 Demo Files Included

### Complex Real-World Examples

**1. complex_sales_analysis.xlsx** (6 sheets, 500 rows)
- Sales transactions across 5 regions
- Product performance analytics
- Time series trends
- Anomaly detection

**2. financial_model.xlsx** (4 sheets)
- Income statement (5-year)
- Balance sheet
- Cash flow statement
- Key financial ratios

**3. customer_churn_analysis.ipynb** (200+ lines)
- 10,000 customer dataset
- Complete ML workflow
- Random Forest model (84.7% accuracy)
- Business recommendations

**4. stock_forecasting.ipynb**
- Time series analysis
- ARIMA modeling
- Forecasting with metrics

---

## 🎬 How to Use

### 1. Upload Your Document
- Click "Upload & Analyze"
- Select `.ipynb` or `.xlsx` file
- Wait 2-5 seconds for processing

### 2. Generate Key Insights (Recommended)
- Switch to "Key Points" tab
- Click "Generate Key Insights"
- Wait 10-30 seconds for AI analysis

### 3. Ask Questions
- Type in the chat interface
- Get instant AI-powered answers
- Follow-up questions maintain context

### 4. Example Questions
```
- "What is this document about?"
- "What are the key findings?"
- "How was [metric] calculated?"
- "Why was [data] removed?"
- "What are the business recommendations?"
- "Are there any data quality issues?"
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Gradio Web UI                      │
│  ┌──────────────┐        ┌──────────────────────┐  │
│  │   Document   │        │    Q&A Interface     │  │
│  │    Viewer    │        │   (with context)     │  │
│  └──────────────┘        └──────────────────────┘  │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│              Context Thread Builder                  │
│  • Parses notebooks/Excel                           │
│  • Extracts cells and dependencies                  │
│  • Infers intents (data loading, modeling, etc.)    │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│           FAISS Vector Indexing                      │
│  • Embeds cell content                              │
│  • Enables semantic search                          │
│  • Fast retrieval (< 100ms)                         │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│            Groq LLM Reasoning                        │
│  • llama-3.3-70b-versatile                          │
│  • Conversation history integration                 │
│  • Citation extraction                              │
│  • Hallucination detection                          │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Upload Processing | < 2 seconds |
| Document Indexing | < 5 seconds |
| Query Response | 2-4 seconds |
| Key Points Generation | 10-30 seconds |
| Groq Inference | < 3 seconds |
| Context Window | 8K tokens |

---

## 🛠️ Technology Stack

- **Frontend:** Gradio (web UI framework)
- **AI/LLM:** Groq API (llama-3.3-70b-versatile)
- **Vector Search:** FAISS (Facebook AI Similarity Search)
- **Data Processing:** Pandas, NumPy
- **Notebook Parsing:** nbformat
- **Excel Handling:** openpyxl, xlsxwriter

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Complete setup guide
- **[MAJOR_UPDATES.md](MAJOR_UPDATES.md)** - Detailed feature documentation
- **[design.md](design.md)** - System architecture
- **[HF_DEPLOYMENT_GUIDE.md](HF_DEPLOYMENT_GUIDE.md)** - Deployment instructions

---

## 🎯 Project Structure

```
context-thread-agent/
├── ui/
│   └── app.py              # Gradio web interface (enhanced)
├── src/
│   ├── groq_integration.py # Groq LLM integration (optimized)
│   ├── reasoning.py        # Answer generation with context
│   ├── retrieval.py        # Vector search engine
│   ├── indexing.py         # FAISS indexing
│   ├── parser.py           # Notebook/Excel parsing
│   ├── dependencies.py     # Context thread building
│   └── intent.py           # Intent classification
├── demo_files/             # Complex demo notebooks & Excel
├── generate_demo_files.py  # Demo file generator
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

---

## 🚦 What's New in This Version

### Major UI/UX Overhaul ✨
- ✅ Professional homepage with clear value proposition
- ✅ Split-screen workspace (viewer + Q&A)
- ✅ Tabbed document viewer
- ✅ Chatbot-style conversation interface
- ✅ Loading indicators and status updates

### Enhanced AI Capabilities 🤖
- ✅ Conversation context maintained across questions
- ✅ Improved Groq prompting for better accuracy
- ✅ Key insights generation feature
- ✅ Higher token limits (2000 tokens)
- ✅ Better citation extraction

### Professional Demo Files 📁
- ✅ Complex sales analysis (500+ rows, 6 sheets)
- ✅ Financial modeling workbook (4 statements)
- ✅ ML notebook (200+ lines, real analysis)
- ✅ Time series forecasting notebook

See [MAJOR_UPDATES.md](MAJOR_UPDATES.md) for complete details.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional file format support (CSV, JSON, etc.)
- More visualization options
- Export functionality for insights
- Multi-language support
- Advanced filtering and search

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **Groq** for lightning-fast LLM inference
- **Gradio** for the intuitive web framework
- **FAISS** for efficient vector search
- **Open-source community** for excellent tools

---

## 📞 Support

- **Issues:** Open a GitHub issue
- **Questions:** Check [QUICKSTART.md](QUICKSTART.md) first
- **Demos:** Try the included demo files

---

## 🎉 Get Started Now!

```bash
git clone https://github.com/Mozzicato/context-thread-agent.git
cd context-thread-agent
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"
python generate_demo_files.py
python main.py ui
```

**Upload a document and start asking questions!** 🚀

---

Made with ❤️ by the Context Thread Agent team