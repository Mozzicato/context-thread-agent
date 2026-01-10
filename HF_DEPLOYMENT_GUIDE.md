# 🚀 Hugging Face Spaces Deployment Guide

## Quick Deploy (5 minutes)

### Option 1: Using Hugging Face Web UI (Easiest)

1. **Create a new Space**
   - Go to https://huggingface.co/new-space
   - Name: `context-thread-agent`
   - License: `mit`
   - Space SDK: **Gradio**
   - Visibility: **Public**
   - Hardware: **CPU Basic** (or GPU if you need it)

2. **Clone the space repo**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/context-thread-agent
   cd context-thread-agent
   ```

3. **Copy files from this repo**
   ```bash
   # From your local context-thread-agent directory, copy everything needed:
   cp -r src/* ../YOUR_SPACE_NAME/src/
   cp -r ui/* ../YOUR_SPACE_NAME/ui/
   cp -r demo_files/* ../YOUR_SPACE_NAME/demo_files/
   cp main.py requirements.txt app.py ../YOUR_SPACE_NAME/
   cp .hfignore ../YOUR_SPACE_NAME/
   ```

4. **Set Secrets in Hugging Face**
   - Go to Space Settings → Secrets and variables
   - Add:
     ```
     GROQ_API_KEY: YOUR_GROQ_API_KEY_HERE
     ```
   - **⚠️ IMPORTANT:** Do NOT commit `.env` file - use Space secrets instead!

5. **Commit and push**
   ```bash
   cd ../YOUR_SPACE_NAME
   git add .
   git commit -m "Deploy Context Thread Agent"
   git push
   ```

6. **Your app will be live at:** `https://huggingface.co/spaces/YOUR_USERNAME/context-thread-agent`

---

### Option 2: Using Hugging Face CLI

```bash
# Install CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create space
huggingface-cli repo create context-thread-agent --type space --space-sdk gradio

# Clone and setup (same as steps 2-5 above)
```

---

## ⚙️ Configuration

### Required Secrets
Set these in your Space → Settings → Secrets and variables:

- `GROQ_API_KEY`: Your Groq API key (required for AI features)

### Optional Secrets
- `OPENAI_API_KEY`: Fallback if Groq is unavailable
- `HUGGINGFACE_HUB_TOKEN`: For downloading models

### Hardware Requirements
- **CPU Basic**: Sufficient for most notebooks
- **CPU Upgrade**: For very large notebooks (>100MB)
- **GPU**: Only if you add custom ML models

---

## 🔧 Troubleshooting

### App won't start
- Check Space logs in Settings → Running
- Ensure all files are committed (especially `app.py`)
- Verify secrets are set correctly

### Import errors
- Make sure `src/` and `ui/` directories are copied
- Check `requirements.txt` has all dependencies

### Groq not working
- Verify `GROQ_API_KEY` secret is set in Space settings
- Check Space logs for API errors

---

## 📁 File Structure for Spaces

```
your-space-repo/
├── app.py              # Entry point (runs the Gradio app)
├── requirements.txt    # Python dependencies
├── src/               # Core logic
├── ui/                # Gradio interface
├── demo_files/        # Sample notebooks
├── .hfignore         # Exclude unnecessary files
└── README.md         # Space description
```

**Note:** `.env` file is NOT included - secrets go in Space settings!
   git add -A
   git commit -m "Deploy Phase 3 Context Thread Agent"
   git push
   ```

✅ **Space should deploy automatically in 2-3 minutes**

---

### Option 2: Using Git (More Control)

```bash
# Inside the cloned space repo
cd /path/to/huggingface/space

# Copy entire src/, ui/, and data directories
cp -r /path/to/context-thread-agent/src .
cp -r /path/to/context-thread-agent/ui .
mkdir -p data/sample_notebooks
cp /path/to/context-thread-agent/data/sample_notebooks/* data/sample_notebooks/

# Copy core files
cp /path/to/context-thread-agent/main.py .
cp /path/to/context-thread-agent/requirements.txt .

# Create app.py (entry point for HF Spaces)
cat > app.py << 'EOF'
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ui.app import NotebookAgentUI

# Create and launch UI
app = NotebookAgentUI()
interface = app.create_interface()
interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
EOF

chmod +x app.py

# Add secrets via git
# (You'll do this in HF web UI: Settings → Secrets)

# Commit
git add -A
git commit -m "Deploy Phase 3 - Notebook Context Agent"
git push

echo "✅ Deployment initiated! Check https://huggingface.co/spaces/YOUR_USERNAME/context-thread-agent"
```

---

## 🔐 Hugging Face Secrets Setup

**Via Web UI (Recommended):**
1. Open your Space: https://huggingface.co/spaces/YOUR_USERNAME/context-thread-agent
2. Click **Settings** (gear icon)
3. Go to **Secrets and variables**
4. Add each secret:
   - **GROQ_API_KEY**: `YOUR_GROQ_API_KEY_HERE`
   - **OPENAI_API_KEY**: (optional, if you have)
   - **HF_TOKEN**: `YOUR_HF_TOKEN_HERE`
5. Click **Save**

**Secrets will be automatically injected as environment variables when the space runs.**

---

## ✅ Verification Checklist

After deployment, verify everything works:

### 1. **Space is Running**
- [ ] Space status shows "Running" (green)
- [ ] No errors in logs (Space runtime tab)

### 2. **UI Loads**
- [ ] Open space URL
- [ ] Gradio interface appears
- [ ] 3 tabs visible (Notebook, Ask Questions, About)

### 3. **Upload Functionality**
- [ ] Can upload a sample notebook
- [ ] Notebook info displays correctly
- [ ] Dependency analysis shows (if available)

### 4. **Query Functionality**
- [ ] Can type a question
- [ ] System returns an answer
- [ ] Citations are displayed (if found)
- [ ] Confidence score is shown

### 5. **Performance**
- [ ] Notebook upload takes <2 seconds
- [ ] Query response takes <10 seconds
- [ ] No timeout errors

---

## 📊 Space Configuration

### Resource Requirements
- **CPU:** 1 vCPU (sufficient)
- **RAM:** 2 GB (sufficient for most notebooks)
- **Storage:** 1 GB (for caching)
- **Startup time:** ~30 seconds
- **Recommended upgrade:** CPU upgrade if <5 second queries needed

### Environment Variables Used
```python
GROQ_API_KEY        # Required for reasoning
OPENAI_API_KEY      # Optional (fallback)
HF_TOKEN            # Optional (for HF integrations)
FAISS_INDEX_PATH    # Defaults to ./data/faiss_index.pkl
METADATA_DB_PATH    # Defaults to ./data/metadata.db
INTENT_CACHE_PATH   # Defaults to ./data/intent_cache.json
DEBUG               # Optional (set to true for verbose logging)
```

---

## 🔄 Updating the Space

To update after making changes:

```bash
# In the cloned space directory
cd /path/to/huggingface/space

# Pull latest
git pull

# Copy updated files
cp -r /path/to/context-thread-agent/src .
cp -r /path/to/context-thread-agent/ui .
cp /path/to/context-thread-agent/main.py .
cp /path/to/context-thread-agent/requirements.txt .

# Commit
git add -A
git commit -m "Update: [your changes]"
git push

# Space will auto-redeploy in ~2 minutes
```

---

## 🐛 Troubleshooting

### Space won't start
- **Check logs:** Space → Logs tab
- **Common issues:**
  - Missing `app.py` entry point
  - Syntax errors in requirements.txt
  - Missing secret variables
  - Port binding (should use 7860 or 0.0.0.0:7860)

### Upload fails
- **Check:** File is valid .ipynb (JSON format)
- **Check:** File size <100MB
- **Check:** Browser console for errors (F12 → Console)

### Queries timeout
- **Cause:** Groq API rate limit or no internet
- **Fix:** Check API key is valid in Secrets
- **Fallback:** Will use mock embeddings (slower but works)

### Citations not showing
- **Note:** Normal with mock embeddings
- **For real embeddings:** Add OPENAI_API_KEY to Secrets
- **Or:** Groq API may need adjustment for embedding generation

### Space runs out of memory
- **Solution:** Upgrade to CPU upgrade in Space settings
- **Or:** Limit notebook size in UI validation
- **Or:** Implement notebook chunking strategy

---

## 📈 Monitoring & Analytics

### View Space Metrics
1. Open Space Settings
2. Go to "Monitoring" tab
3. Check:
   - **CPU usage**
   - **RAM usage**
   - **Request count**
   - **Error rate**
   - **Average response time**

### Optimize Based on Usage
- If CPU >80%: Recommend CPU upgrade
- If RAM >90%: Reduce context window size
- If errors >5%: Check API keys/connectivity

---

## 🎯 Next Steps After Deployment

### 1. **Share the Space**
```
Public URL: https://huggingface.co/spaces/YOUR_USERNAME/context-thread-agent
Share via:
- LinkedIn
- Twitter
- Email
- GitHub
- Dev communities (Reddit, HackerNews, ProductHunt)
```

### 2. **Optimize Based on Usage**
- Monitor error logs
- Gather user feedback
- Adjust retrieval parameters
- Add more sample notebooks

### 3. **Enhance Features**
- Add change impact analysis
- Implement notebook comparison
- Add visualization dashboard
- Create API endpoint for programmatic access

### 4. **Scale if Needed**
- Switch to larger model (if needed)
- Implement request queueing
- Add caching layer
- Consider vector DB for larger notebook collections

---

## 🔗 Useful Links

- **Hugging Face Spaces:** https://huggingface.co/spaces
- **Create New Space:** https://huggingface.co/new-space
- **Space Documentation:** https://huggingface.co/docs/hub/spaces-overview
- **Gradio Docs:** https://www.gradio.app/
- **API Keys:**
  - Groq: https://console.groq.com/keys
  - OpenAI: https://platform.openai.com/account/api-keys
  - Hugging Face: https://huggingface.co/settings/tokens

---

## ✨ Summary

**Deployment time:** 5-10 minutes  
**Complexity:** Easy (point-and-click through web UI)  
**Maintenance:** Minimal (GitHub integration auto-deploys on push)  
**Cost:** Free tier includes:
- 0.5 CPU
- 2GB RAM
- Public hosting
- Auto-update on git push

**Recommended:** Start with free tier, upgrade to paid CPU if needed for better performance.

---

**Generated:** 2024-01-06  
**Status:** Ready for immediate deployment ✅
