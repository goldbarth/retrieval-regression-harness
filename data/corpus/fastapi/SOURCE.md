# Corpus source

| | |
|---|---|
| Repository | https://github.com/fastapi/fastapi |
| Commit | `c3f316b7e814667e8ee81e03a7330d00ee61e45c` |
| Vendored at | 2026-08-20 |
| Documents | 24 |
| License | MIT, copied verbatim to `LICENSE` next to this file |

Reproduce with:

```
git clone https://github.com/fastapi/fastapi /tmp/fastapi && git -C /tmp/fastapi checkout c3f316b7e814667e8ee81e03a7330d00ee61e45c
uv run python scripts/vendor_corpus.py --checkout /tmp/fastapi
```

Selected from `docs/en/docs`, listed in `scripts/vendor_corpus.py`.
Not a verbatim copy: `{* ... *}` includes are resolved into fenced code
blocks, because the code examples live in `docs_src/` and a corpus without them
would never exercise retrieval over code. The commit plus this script is what
makes the copy checkable.
