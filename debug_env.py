#!/usr/bin/env python3
from pathlib import Path

env_path = Path('.env')
print(f"Checking {env_path.absolute()}")
print(f"File exists: {env_path.exists()}")
print(f"File size: {env_path.stat().st_size if env_path.exists() else 'N/A'}")

if env_path.exists():
    content = env_path.read_bytes()
    print(f"Raw bytes (first 200): {content[:200]}")
    print(f"\nDecoded content (first 500 chars):")
    try:
        text = env_path.read_text(encoding='utf-8')
        print(repr(text[:500]))
        print(f"\n\nLines containing GROQ:")
        for i, line in enumerate(text.splitlines(), 1):
            if 'GROQ' in line:
                print(f"  Line {i}: {repr(line)}")
    except Exception as e:
        print(f"Error reading: {e}")
