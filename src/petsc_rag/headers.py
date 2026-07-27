"""Ground-truth PETSc function signatures, extracted from $PETSC_DIR headers.

Companion to the tutorial-text RAG in retrieve.py. The tutorial RAG helps the
LLM pick a *pattern*; this index helps it call functions with the *right
arity*. The motivating failure mode: the LLM emits
`DMPlexSetSNESLocalFEM(dm, NULL, NULL, NULL)` (4 args) when the real signature
is `DMPlexSetSNESLocalFEM(DM, PetscBool, void *)` (3 args). The compiler error
names the symbol; we look it up here and feed the canonical signature back
into the fix-it turn.

This is *not* problem-specific: any PETSc API call the agent gets wrong can be
corrected by the same loop, because the index covers the entire installed
public header surface.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable


_SIG_RE = re.compile(
    r"^PETSC_EXTERN\s+PetscErrorCode\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;]*)\)\s*;",
    re.MULTILINE,
)

_state: dict[str, object] = {"loaded": False}


def _petsc_include_dirs() -> list[Path]:
    root = os.environ.get("PETSC_DIR")
    if not root:
        return []
    inc = Path(root) / "include"
    if not inc.is_dir():
        return []
    return [inc]


def _walk_headers(roots: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        for p in root.rglob("*.h"):
            # Skip Fortran shims — they redeclare the same symbols with
            # different signatures and would shadow the C ones.
            if "/finclude/" in str(p):
                continue
            out.append(p)
    return out


def _parse_header(path: Path) -> list[tuple[str, str, str]]:
    """Return [(name, full_signature_line, header_basename)] for one file."""
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return []
    out: list[tuple[str, str, str]] = []
    for m in _SIG_RE.finditer(text):
        name = m.group(1)
        args = m.group(2).strip()
        sig = f"PetscErrorCode {name}({args});"
        out.append((name, sig, path.name))
    return out


def _load() -> None:
    if _state.get("loaded"):
        return
    index: dict[str, tuple[str, str]] = {}
    for header in _walk_headers(_petsc_include_dirs()):
        for name, sig, hdr in _parse_header(header):
            # First-seen wins. Some symbols appear in multiple headers via
            # forward-decls; the canonical header is usually the first
            # alphabetical, which is good enough for our purpose.
            if name not in index:
                index[name] = (sig, hdr)
    _state["index"] = index
    _state["loaded"] = True


def lookup(name: str) -> tuple[str, str] | None:
    """Return (signature, header_basename) for a PETSc function name, or None."""
    _load()
    idx = _state.get("index") or {}
    return idx.get(name)  # type: ignore[return-value]


def size() -> int:
    _load()
    idx = _state.get("index") or {}
    return len(idx)  # type: ignore[arg-type]


# Tokens that look like PETSc symbols but aren't worth chasing.
_SKIP_TOKENS = {
    "PETSC_NULL", "PETSC_TRUE", "PETSC_FALSE", "PETSC_COMM_WORLD",
    "PETSC_COMM_SELF", "PETSC_DECIDE", "PETSC_DETERMINE", "PETSC_DEFAULT",
    "PETSC_ERR_ARG_WRONG", "PETSC_ERR_ARG_OUTOFRANGE", "PETSC_ERR_SUP",
    "PETSC_ERR_PLIB", "PETSC_ERR_MEMC",
    "MPI_COMM_WORLD", "MPI_COMM_SELF",
    "PetscErrorCode", "PetscInt", "PetscReal", "PetscScalar", "PetscBool",
    "PetscMPIInt", "PetscObject", "PetscFunctionBegin", "PetscFunctionReturn",
    "PetscCall", "PetscCallMPI", "PetscCheck", "SETERRQ",
    "DM", "TS", "SNES", "KSP", "PC", "Mat", "Vec", "IS", "AO", "PetscFE",
    "PetscDS", "PetscFV", "PetscQuadrature", "PetscWeakForm",
}

_SYMBOL_RE = re.compile(r"\b([A-Z][A-Za-z][A-Za-z0-9_]{2,})\b")


def extract_petsc_symbols(text: str, limit: int = 8) -> list[str]:
    """Pull plausible PETSc function names out of compiler/runtime output.

    Looks for tokens matching ^[A-Z][A-Za-z]\\w{2,}$ that ALSO exist in the
    header index. Anything that doesn't appear in the index is dropped — so
    user-defined identifiers, MPI calls, etc. don't pollute the result.
    """
    _load()
    idx = _state.get("index") or {}
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in _SYMBOL_RE.finditer(text or ""):
        tok = m.group(1)
        if tok in _SKIP_TOKENS or tok in seen_set:
            continue
        if tok in idx:  # type: ignore[operator]
            seen.append(tok)
            seen_set.add(tok)
            if len(seen) >= limit:
                break
    return seen


def format_signature_block(names: list[str]) -> str:
    """Render a fix-it-turn block of canonical signatures for the given names."""
    if not names:
        return ""
    lines = ["Real PETSc signatures (from installed headers) for symbols in the error:"]
    for n in names:
        hit = lookup(n)
        if hit is None:
            continue
        sig, hdr = hit
        lines.append(f"  // from <{hdr}>")
        lines.append(f"  {sig}")
    if len(lines) == 1:
        return ""
    lines.append("Match argument count and types EXACTLY. If your call differs, fix it.")
    return "\n".join(lines)
