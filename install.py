"""
Run this once before starting the system: python install.py
"""
import subprocess
import sys


def run(cmd):
    print(f"  >> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [WARN] command returned {result.returncode}")
    return result.returncode == 0


def main():
    pip = [sys.executable, "-m", "pip", "install"]

    print("\n=== Step 1: Core packages ===")
    packages = [
        "neo4j>=5.0",
        "python-binance>=1.0.19",
        "feedparser>=6.0",
        "transformers>=4.40",
        "sentence-transformers>=2.6",
        "hmmlearn>=0.3",
        "scikit-learn>=1.3",
        "openai>=1.30",
        "schedule>=1.2",
        "requests>=2.31",
        "accelerate",
    ]
    run(pip + packages)

    print("\n=== Step 2: spaCy + language model ===")
    run(pip + ["spacy>=3.7"])
    run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

    print("\n=== Step 3: HDBSCAN ===")
    # hdbscan may need C compiler; try binary wheel first
    ok = run(pip + ["hdbscan>=0.8", "--prefer-binary"])
    if not ok:
        print("  [WARN] hdbscan failed. Install manually with: pip install hdbscan --prefer-binary")

    print("\n=== Step 4: PyTorch Geometric ===")
    # Base package works without C++ extensions on any PyTorch version
    run(pip + ["torch-geometric>=2.5"])
    # Try extension wheels matching your local torch build
    import torch
    torch_ver = torch.__version__.split("+")[0]
    cuda_tag = "cu" + torch.version.cuda.replace(".", "") if torch.cuda.is_available() else "cpu"
    pyg_url = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_tag}.html"
    print(f"  Attempting PyG extensions from: {pyg_url}")
    ok = run(pip + [
        "pyg-lib", "torch-scatter", "torch-sparse",
        "torch-cluster", "torch-spline-conv",
        "-f", pyg_url, "--no-index"
    ])
    if not ok:
        print("  [INFO] PyG C++ extensions not available for your nightly build.")
        print("         The system will run in pure-PyTorch mode (slightly slower but fully functional).")

    print("\n=== All done! ===")
    print("Next steps:")
    print("  1. Copy .env.example -> .env and fill in your credentials")
    print("  2. Start Neo4j Desktop and create a database")
    print("  3. Run: python main.py")


if __name__ == "__main__":
    main()
