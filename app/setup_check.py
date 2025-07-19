"""
Script για να ελέγξει αν όλα είναι έτοιμα για το RAG system.

Τρέξε το με: python setup_check.py
"""

import subprocess
import sys
import os
import requests


def check_python_version():
    """Έλεγχος Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} - Need 3.8+")
        return False


def check_dependencies():
    """Έλεγχος αν όλα τα packages είναι εγκατεστημένα."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn", 
        "sqlalchemy",
        "chromadb",
        "sklearn",
        "numpy",
        "ollama"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"❌ {package} - NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def check_ollama():
    """Έλεγχος αν το Ollama τρέχει."""
    print("\n🦙 Checking Ollama...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            print("✅ Ollama is running")
            
            # Έλεγχος για models
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            print(f"   Found {len(models)} models: {', '.join(model_names)}")
            
            # Έλεγχος για τα απαραίτητα models
            required_models = ["mistral", "nomic-embed-text"]
            missing_models = []
            
            for required in required_models:
                # Check if any version of the model exists (with or without :latest)
                if not any(required in model for model in model_names):
                    missing_models.append(required)
            
            if missing_models:
                print(f"\n⚠️  Missing models: {', '.join(missing_models)}")
                for model in missing_models:
                    print(f"   Run: ollama pull {model}")
                return False
            else:
                print("✅ All required models are available")
                return True
        else:
            print("❌ Ollama is not responding properly")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running")
        print("   Run: ollama serve")
        return False


def check_knowledge_base():
    """Έλεγχος αν υπάρχει το knowledge base."""
    print("\n📚 Checking knowledge base...")
    
    kb_path = "data/knowledge_base.txt"
    if os.path.exists(kb_path):
        size = os.path.getsize(kb_path)
        print(f"✅ Knowledge base found ({size} bytes)")
        return True
    else:
        print(f"❌ Knowledge base not found at {kb_path}")
        print("   Make sure you have created data/knowledge_base.txt")
        return False


def check_project_structure():
    """Έλεγχος project structure."""
    print("\n📁 Checking project structure...")
    
    required_files = [
        "app/__init__.py",
        "app/main.py",
        "app/routes.py",
        "app/models.py",
        "app/database.py",
        "app/schemas.py",
        "app/llm_service.py",
        "app/kb_parser.py",
        "app/embeddings_service.py",
        "app/chromadb_service.py",
        "app/tfidf_service.py",
        "app/rag_service.py",
        "data/knowledge_base.txt"
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - NOT found")
            missing.append(file)
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} files")
        return False
    
    return True


def main():
    """Run all checks."""
    print("🔍 RAG System Setup Checker")
    print("=" * 50)
    
    all_ok = True
    
    all_ok &= check_python_version()
    all_ok &= check_dependencies()
    all_ok &= check_ollama()
    all_ok &= check_knowledge_base()
    all_ok &= check_project_structure()
    
    print("\n" + "=" * 50)
    
    if all_ok:
        print("✅ Everything is ready! You can now run:")
        print("   1. uvicorn app.main:app --reload")
        print("   2. python test_rag.py")
    else:
        print("❌ Some issues need to be fixed before running the system")
        print("   Fix the issues above and run this script again")


if __name__ == "__main__":
    main()