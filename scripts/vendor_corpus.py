"""Vendor a subset of the FastAPI documentation as the retrieval corpus.

The corpus is committed, not fetched at run time, so a run never depends on the
network or on upstream edits. This script exists to make that copy reproducible:
given the same checkout, it rewrites the same files.

Usage:
    uv run python scripts/vendor_corpus.py --checkout /path/to/fastapi
"""

import argparse
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO = "https://github.com/fastapi/fastapi"
LICENSE = "MIT"

CORPUS_ROOT = Path("data/corpus/fastapi")

# Paths inside the checkout. Includes resolve against the mkdocs config
# directory, which is DOCS_BASE, not against the file that carries them.
DOCS_BASE = Path("docs/en")
DOCS_DIR = DOCS_BASE / "docs"

# Chosen against the three failure modes the gold questions have to provoke.
# reference/ is deliberately absent: those files are mkdocstrings stubs
# (`::: fastapi.Depends`) whose content only appears when the docs are built,
# and their headings carry no explicit anchor.
DOCUMENTS: tuple[str, ...] = (
    # Near duplicates: tutorial and advanced describe the same mechanism twice.
    "tutorial/dependencies/index.md",
    "tutorial/dependencies/classes-as-dependencies.md",
    "tutorial/dependencies/sub-dependencies.md",
    "tutorial/dependencies/dependencies-in-path-operation-decorators.md",
    "tutorial/dependencies/global-dependencies.md",
    "tutorial/dependencies/dependencies-with-yield.md",
    "advanced/advanced-dependencies.md",
    "advanced/testing-dependencies.md",
    # Term collision: "response" means the model, the class and the status code.
    "tutorial/response-model.md",
    "tutorial/response-status-code.md",
    "advanced/additional-responses.md",
    "advanced/additional-status-codes.md",
    "advanced/response-directly.md",
    "advanced/custom-response.md",
    "advanced/response-change-status-code.md",
    "advanced/response-headers.md",
    "advanced/response-cookies.md",
    # Term collision: "request" means the body, the form and the Request object.
    "tutorial/request-files.md",
    "tutorial/request-forms.md",
    "tutorial/request-forms-and-files.md",
    "advanced/using-request-directly.md",
    # Multi section: the concept and its parameters sit in different sections.
    "tutorial/background-tasks.md",
    "tutorial/middleware.md",
    "advanced/middleware.md",
)

# `{* path ln[a:b] hl[...] *}`. Only `ln` selects content, `hl` is a highlight
# hint for the rendered page and carries nothing a retriever could use.
INCLUDE_PATTERN = re.compile(
    r"^\{\*\s*(?P<path>[\w./-]+\.\w+)(?P<options>[^*]*)\*\}\s*$"
)
LINE_RANGE_PATTERN = re.compile(r"ln\[(?P<start>\d+):(?P<end>\d+)\]")

FENCE_LANGUAGES = {".py": "python"}


def resolve_include(source: Path, options: str) -> str:
    """Return the referenced source file as a fenced code block."""
    lines = source.read_text(encoding="utf-8").splitlines()

    line_range = LINE_RANGE_PATTERN.search(options)
    if line_range is not None:
        # 1-based and inclusive on both ends, verified against
        # docs_src/dependencies/tutorial013_an_py310.py ln[19:21].
        start = int(line_range.group("start"))
        end = int(line_range.group("end"))
        lines = lines[start - 1 : end]

    # Fence on the language, not on the file suffix: a ```py block is
    # highlighted by fewer tools than ```python, and the fence is the only
    # hint a later chunker has that this part is code.
    language = FENCE_LANGUAGES.get(source.suffix, source.suffix.lstrip("."))
    body = "\n".join(lines)
    return f"```{language}\n{body}\n```"


def vendor_document(relative_path: str, checkout: Path, corpus_root: Path) -> int:
    """Copy one document, resolving its includes. Returns the include count."""
    source = checkout / DOCS_DIR / relative_path
    resolved: list[str] = []
    include_count = 0

    for line in source.read_text(encoding="utf-8").splitlines():
        match = INCLUDE_PATTERN.match(line)
        if match is None:
            resolved.append(line)
            continue

        included = (checkout / DOCS_BASE / match.group("path")).resolve()
        if not included.is_file():
            raise FileNotFoundError(f"{relative_path}: include {included} is missing")

        resolved.append(resolve_include(included, match.group("options")))
        include_count += 1

    target = corpus_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(resolved) + "\n", encoding="utf-8")
    return include_count


def read_commit(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def write_source_note(corpus_root: Path, commit: str, documents: int) -> None:
    vendored_at = datetime.now(UTC).date().isoformat()
    note = f"""# Corpus source

| | |
|---|---|
| Repository | {REPO} |
| Commit | `{commit}` |
| Vendored at | {vendored_at} |
| Documents | {documents} |
| License | {LICENSE}, copied verbatim to `LICENSE` next to this file |

Reproduce with:

```
git clone {REPO} /tmp/fastapi && git -C /tmp/fastapi checkout {commit}
uv run python scripts/vendor_corpus.py --checkout /tmp/fastapi
```

Selected from `{DOCS_DIR}`, listed in `scripts/vendor_corpus.py`.
Not a verbatim copy: `{{* ... *}}` includes are resolved into fenced code
blocks, because the code examples live in `docs_src/` and a corpus without them
would never exercise retrieval over code. The commit plus this script is what
makes the copy checkable.
"""
    (corpus_root / "SOURCE.md").write_text(note, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    args = parser.parse_args()

    checkout: Path = args.checkout
    corpus_root: Path = args.corpus_root

    includes = 0
    for relative_path in DOCUMENTS:
        includes += vendor_document(relative_path, checkout, corpus_root)

    commit = read_commit(checkout)
    write_source_note(corpus_root, commit, len(DOCUMENTS))
    print(f"{len(DOCUMENTS)} documents, {includes} includes resolved, at {commit[:8]}")


if __name__ == "__main__":
    main()
