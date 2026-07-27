"""Build a FAISS index over the PETSc tutorial corpus.

Run once. Walks the PETSc tutorial tree, embeds each .c file with a local
SentenceTransformer model, and writes a FAISS index + metadata pickle that
retrieve.py loads at query time.

Usage:
    python -m petsc_rag.build_index
    python -m petsc_rag.build_index --petsc-root /path/to/petsc --out src/petsc_rag/index
"""

import argparse
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_PETSC_ROOT = Path(
    "/opt/homebrew/Cellar/petsc/3.24.6/share/petsc/examples/src"
)
DEFAULT_OUT = Path(__file__).parent / "index"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
MAX_CHARS = 8000  # truncate very long tutorials; most are well under this


def collect_tutorials(petsc_root: Path) -> list[tuple[Path, str, str]]:
    """Return [(absolute_path, relative_path, subsystem)] for every tutorial .c."""
    out = []
    for c_file in sorted(petsc_root.rglob("tutorials/ex*.c")):
        rel = c_file.relative_to(petsc_root)
        # rel looks like "ksp/tutorials/ex2.c" -> subsystem = "ksp"
        subsystem = rel.parts[0] if rel.parts else "unknown"
        out.append((c_file, str(rel), subsystem))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--petsc-root", type=Path, default=DEFAULT_PETSC_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    if not args.petsc_root.exists():
        raise SystemExit(f"PETSc tutorial root not found: {args.petsc_root}")

    args.out.mkdir(parents=True, exist_ok=True)

    tutorials = collect_tutorials(args.petsc_root)
    print(f"Found {len(tutorials)} tutorials under {args.petsc_root}")

    chunks: list[str] = []
    metas: list[dict] = []
    for abs_path, rel_path, subsystem in tutorials:
        try:
            text = abs_path.read_text(errors="ignore")
        except OSError as e:
            print(f"  skip {rel_path}: {e}")
            continue
        chunks.append(text[:MAX_CHARS])
        metas.append({"path": rel_path, "subsystem": subsystem, "chars": len(text)})

    print(f"Embedding {len(chunks)} chunks with {args.model}...")
    model = SentenceTransformer(args.model)
    vectors = model.encode(
        chunks,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss_path = args.out / "faiss.bin"
    store_path = args.out / "store.pkl"
    faiss.write_index(index, str(faiss_path))
    with open(store_path, "wb") as f:
        pickle.dump(
            {"chunks": chunks, "metas": metas, "model_name": args.model},
            f,
        )

    print(f"Wrote {faiss_path} ({vectors.shape[0]} vectors, dim={vectors.shape[1]})")
    print(f"Wrote {store_path}")


if __name__ == "__main__":
    main()
