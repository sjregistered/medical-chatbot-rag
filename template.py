"""
template.py – Medical Chatbot Project Structure Generator
As specified in the PDF: modular backend structure using template.py

This script creates the full project directory structure when run standalone.
It ensures all necessary directories and placeholder files exist.
"""

import os

# Project directory structure definition
PROJECT_STRUCTURE = {
    "src": {
        "__init__.py": "# Medical Chatbot Source Package",
        "ingestion.py": None,      # Core module – not overwritten
        "embeddings.py": None,     # Core module – not overwritten
        "vector_store.py": None,   # Core module – not overwritten
        "llm.py": None,            # Core module – not overwritten
        "chat.py": None,           # Core module – not overwritten
    },
    "data": {
        "medical_knowledge.txt": None,  # Medical corpus – not overwritten
    },
    "templates": {
        "index.html": None,        # Frontend template – not overwritten
    },
    "static": {
        "css": {
            "style.css": None,     # Styles – not overwritten
        },
        "js": {
            "app.js": None,        # Frontend logic – not overwritten
        },
    },
    "chroma_db": {},                # Vector store persistence directory
}


def create_structure(base_path: str = ".", structure: dict = None) -> None:
    """
    Recursively create the project directory structure.
    
    Args:
        base_path: Root directory for the project.
        structure: Dictionary defining the directory tree.
    """
    if structure is None:
        structure = PROJECT_STRUCTURE
    
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        
        if isinstance(content, dict):
            # It's a directory
            os.makedirs(path, exist_ok=True)
            print(f"  📁 {path}/")
            create_structure(path, content)
        elif content is not None:
            # It's a file with placeholder content (don't overwrite existing)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"  📄 {path} (created)")
            else:
                print(f"  ✓  {path} (exists)")
        else:
            # None means the file should exist but we don't create it
            if os.path.exists(path):
                print(f"  ✓  {path} (exists)")
            else:
                print(f"  ⚠  {path} (missing – create manually)")


def verify_structure(base_path: str = ".") -> dict:
    """
    Verify the project structure and return a status report.
    
    Args:
        base_path: Root directory to verify.
    
    Returns:
        Dictionary with verification results.
    """
    required_files = [
        "app.py",
        "template.py",
        "requirements.txt",
        "README.md",
        "src/__init__.py",
        "src/ingestion.py",
        "src/embeddings.py",
        "src/vector_store.py",
        "src/llm.py",
        "src/chat.py",
        "data/medical_knowledge.txt",
        "templates/index.html",
        "static/css/style.css",
        "static/js/app.js",
    ]
    
    results = {"present": [], "missing": []}
    
    for filepath in required_files:
        full_path = os.path.join(base_path, filepath)
        if os.path.exists(full_path):
            results["present"].append(filepath)
        else:
            results["missing"].append(filepath)
    
    return results


if __name__ == "__main__":
    print("=" * 50)
    print("Medical Chatbot – Project Structure Generator")
    print("=" * 50)
    print()
    
    create_structure(".")
    
    print()
    print("Verifying structure...")
    status = verify_structure(".")
    
    print(f"\n✅ Present: {len(status['present'])} files")
    if status["missing"]:
        print(f"❌ Missing: {len(status['missing'])} files")
        for f in status["missing"]:
            print(f"   - {f}")
    else:
        print("🎉 All required files present!")
