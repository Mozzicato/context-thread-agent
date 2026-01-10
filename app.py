#!/usr/bin/env python3
"""
Hugging Face Spaces entry point for Context Thread Agent
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the Gradio app
from ui.app import create_gradio_app

if __name__ == "__main__":
    # Create and launch the app
    demo = create_gradio_app()

    # Launch for HF Spaces (they handle port and host)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True
    )