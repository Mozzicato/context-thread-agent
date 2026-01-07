# 🚀 Quick Start Guide - Context Thread Agent

## Complete Setup in 5 Minutes

### Step 1: Get Your Groq API Key (Free) 🔑

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Click "Create API Key"
5. Copy your key (starts with `gsk_...`)

### Step 2: Install Dependencies 📦

```bash
cd /workspaces/context-thread-agent

# Install all required packages
pip install -r requirements.txt
```

### Step 3: Set Your API Key 🔐

```bash
# Linux/Mac
export GROQ_API_KEY="your_groq_key_here"

# Windows CMD
set GROQ_API_KEY=your_groq_key_here

# Windows PowerShell
$env:GROQ_API_KEY="your_groq_key_here"

# Or create a .env file
echo 'GROQ_API_KEY=your_groq_key_here' > .env
```

### Step 4: Generate Demo Files 📁

```bash
# Create complex demo notebooks and Excel files
python generate_demo_files.py
```

This creates:
- `demo_files/complex_sales_analysis.xlsx` - 6 sheets, 500 rows
- `demo_files/financial_model.xlsx` - Complete financial statements
- `demo_files/customer_churn_analysis.ipynb` - ML analysis notebook
- `demo_files/stock_forecasting.ipynb` - Time series forecasting

### Step 5: Launch the Application 🎯

```bash
# Start the web interface
python main.py ui --port 7860

# Or with public sharing (creates shareable link)
python main.py ui --port 7860 --share
```

Open your browser to: **http://localhost:7860**

---

## 📖 How to Use the Application

### Upload & Analyze

1. **Click "Upload & Analyze"** button on the homepage
2. **Select a file:**
   - Jupyter Notebook (`.ipynb`)
   - Excel file (`.xlsx`, `.xls`)
3. **Wait for processing** (2-5 seconds)

### View Your Document

**Left Panel - Document Viewer:**
- **Content Tab**: Browse through cells/sheets
- **Key Points Tab**: Click "Generate Key Insights" for AI summary

### Ask Questions

**Right Panel - Q&A Interface:**
1. Type your question in the text box
2. Click "Ask" or press Enter
3. View the answer with citations
4. Ask follow-up questions (context is maintained!)

---

## 💡 Example Usage Workflow

### Analyzing a Notebook

```
1. Upload: customer_churn_analysis.ipynb
2. Generate Key Insights: Click button, wait 15 seconds
3. Ask: "What is this analysis about?"
   → AI explains it's customer churn prediction
4. Ask: "What were the most important features?"
   → AI lists tenure, contract type, charges with citations
5. Ask: "Why did you choose Random Forest?"
   → AI references the specific cells showing model selection
```

### Analyzing Excel Data

```
1. Upload: complex_sales_analysis.xlsx
2. Browse: Check the "Content" tab to see all sheets
3. Generate Key Insights: Get overview of all data
4. Ask: "What are the sales trends?"
5. Ask: "Which region performs best?"
6. Ask: "Are there any anomalies in the data?"
```

---

## 🎯 Sample Questions to Try

### General Understanding
- "What is this document about?"
- "What are the main objectives?"
- "What data sources are used?"

### Technical Details
- "How was [metric] calculated?"
- "Why was [data] removed/filtered?"
- "What preprocessing steps were taken?"

### Results & Insights
- "What are the key findings?"
- "What patterns were discovered?"
- "What are the recommendations?"

### Data Quality
- "Are there any missing values?"
- "What data quality issues exist?"
- "Were there any anomalies detected?"

---

## 🔧 Troubleshooting

### "GROQ_API_KEY not found"
**Solution:** 
```bash
export GROQ_API_KEY="your_key_here"
# Then restart: python main.py ui
```

### "No module named 'openpyxl'"
**Solution:**
```bash
pip install openpyxl xlsxwriter
```

### "File upload failed"
**Solution:**
- Check file is valid .ipynb or .xlsx
- Ensure file isn't corrupted
- Try a demo file first

### Key Points generation slow
**Normal behavior!** 
- Takes 10-30 seconds for complex documents
- Watch for the loading indicator
- Groq is analyzing the entire document

### Questions get generic answers
**Tips:**
- Be more specific in your questions
- Reference specific parts: "In the sales data..."
- Generate Key Insights first for better context

---

## 🎓 Understanding the Features

### 📚 Citations
Every answer includes references to specific cells/sections:
```
Answer: The model achieved 84.7% accuracy...

References:
1. cell_5 [code] - Model training
2. cell_7 [code] - Performance evaluation
```

### 💬 Conversation Context
The AI remembers previous questions:
```
Q1: "What model was used?"
A1: "Random Forest classifier..."

Q2: "Why that one?" ← AI knows "that one" = Random Forest
A2: "Random Forest was chosen because..."
```

### 🔑 Key Insights Generation
AI analyzes and summarizes:
- Purpose and methodology
- Data characteristics
- Key findings
- Issues or concerns
- Conclusions and recommendations

---

## 📊 Demo File Details

### complex_sales_analysis.xlsx
- **500 sales transactions** across 5 regions
- **8 products** with pricing and discounts
- **Regional summaries** with KPIs
- **Time series** monthly trends
- **Anomaly detection** for unusual transactions

### financial_model.xlsx
- **5-year projections** (3 historical, 2 forecast)
- **Complete financial statements**
- **Key ratios** and metrics
- **Integrated model** (I/S → B/S → C/F)

### customer_churn_analysis.ipynb
- **10,000 customers** with 50+ features
- **Complete ML workflow** from EDA to deployment
- **Random Forest model** with 84.7% accuracy
- **Business recommendations** included

### stock_forecasting.ipynb
- **5 years daily stock data**
- **ARIMA time series modeling**
- **Stationarity testing**
- **Forecast evaluation** with RMSE/MAE

---

## 🚀 Advanced Features

### Custom Analysis
Upload your own notebooks or Excel files!
- Works with any Jupyter notebook
- Supports multi-sheet Excel workbooks
- Handles complex data structures

### Batch Questions
Ask multiple related questions:
1. Overview question
2. Specific methodology question
3. Results interpretation
4. Recommendations

### Export & Share
- Take screenshots of insights
- Copy answers for reports
- Share the public link (`--share` flag)

---

## 📞 Need Help?

1. **Check the demo files first** - They showcase all features
2. **Read MAJOR_UPDATES.md** - Detailed feature documentation
3. **Review error messages** - Usually self-explanatory
4. **Try simpler questions first** - Build up to complex queries

---

## 🎉 You're All Set!

Start with a demo file, generate key insights, and ask questions. The AI will guide you through understanding your document.

**Happy Analyzing! 🚀**
