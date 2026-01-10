"""Space-friendly entrypoint for Context Thread Agent.
"""

from ui.app import create_gradio_app


def main():
    demo = create_gradio_app()
    # Launch on all interfaces so HF Spaces can route to it
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
