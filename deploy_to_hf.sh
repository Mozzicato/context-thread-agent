#!/bin/bash
# Deploy to Hugging Face Spaces script

echo "🚀 Context Thread Agent - Hugging Face Spaces Deployment"
echo ""

# Check if HF CLI is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ Hugging Face CLI not found. Install with: pip install huggingface_hub"
    exit 1
fi

# Check if logged in
if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ Not logged in to Hugging Face. Run: huggingface-cli login"
    exit 1
fi

echo "Enter your Hugging Face username:"
read HF_USER

echo "Enter space name (default: context-thread-agent):"
read SPACE_NAME
SPACE_NAME=${SPACE_NAME:-context-thread-agent}

echo ""
echo "📦 Creating Hugging Face Space: $HF_USER/$SPACE_NAME"
huggingface-cli repo create $SPACE_NAME --type space --space-sdk gradio

echo ""
echo "📋 Next steps:"
echo "1. Clone the space: git clone https://huggingface.co/spaces/$HF_USER/$SPACE_NAME"
echo "2. Copy files: cp -r src ui demo_files main.py requirements.txt app.py .hfignore [space-repo]/"
echo "3. Set GROQ_API_KEY in Space Settings → Secrets"
echo "4. Commit and push: cd [space-repo] && git add . && git commit -m 'Deploy' && git push"
echo ""
echo "✅ Space created! URL: https://huggingface.co/spaces/$HF_USER/$SPACE_NAME"