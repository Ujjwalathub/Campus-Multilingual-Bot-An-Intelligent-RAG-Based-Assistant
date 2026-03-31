"""
Verification script to test LangChain imports after fix
Run this to ensure the ModuleNotFoundError is resolved
"""

print("🔍 Verifying LangChain imports...")
print("-" * 50)

try:
    print("✓ Testing langchain_classic.chains import...")
    from langchain_classic.chains import RetrievalQA
    print("  ✅ SUCCESS: RetrievalQA imported from langchain_classic")
except ImportError as e:
    print(f"  ❌ FAILED: {e}")
    print("  💡 Solution: Run 'pip install langchain-classic'")
    exit(1)

try:
    print("\n✓ Testing langchain_community imports...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    print("  ✅ SUCCESS: Community imports working")
except ImportError as e:
    print(f"  ❌ FAILED: {e}")
    exit(1)

try:
    print("\n✓ Testing langchain.prompts import...")
    from langchain.prompts import PromptTemplate
    print("  ✅ SUCCESS: PromptTemplate imported")
except ImportError as e:
    print(f"  ❌ FAILED: {e}")
    exit(1)

print("\n" + "=" * 50)
print("🎉 ALL IMPORTS SUCCESSFUL!")
print("=" * 50)
print("\nYour RAG pipeline should now work correctly.")
print("Next steps:")
print("  1. Ensure .env file has SARVAM_API_KEY")
print("  2. Run: python ingest.py")
print("  3. Run: streamlit run app.py")
