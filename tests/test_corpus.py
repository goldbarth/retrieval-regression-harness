"""Tests for reading the vendored corpus as sections.

Two kinds of test live here, and they fail for different reasons:
  - tmp_path tests pin the parser's rules against a document written here.
  - the last test reads the real 24 files. It fails when the corpus changes
    under the parser, which is the day every expected_source goes dangling.
"""

from pathlib import Path

import pytest

from harness.core.corpus import CorpusError, iter_sections, read_sections

# Anchored on the test file, not on the working directory: the corpus is repo
# data, and pytest must find it from wherever it is started.
CORPUS_ROOT = Path(__file__).parent.parent / "data" / "corpus" / "fastapi"

_DOCUMENT = """# Title { #title }

Intro line.

## Parent { #parent }

Parent text.

### Child { #child }

Child text.

## Sibling { #sibling }

Sibling text.
"""
"""One document that carries every structural case at once: a top heading, a
nested level, and a sibling that follows a nested block."""


def _write(root: Path, relative_path: str, text: str) -> None:
    """Write a corpus file below root, creating parent directories.

    Python note: `path.parent.mkdir(parents=True, exist_ok=True)` is
    Directory.CreateDirectory - it does not throw when the directory is
    already there. Then `path.write_text(text, encoding="utf-8")`; name the
    encoding, the platform default is not utf-8 everywhere.

    `tmp_path` is a pytest fixture: a fresh Path per test, cleaned up for you.
    """
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_sections_do_not_nest(tmp_path: Path) -> None:
    """A `###` block belongs to itself, not also to the `##` above it.

    Decided: not nesting. Every sentence sits in exactly one section, so a
    later recall number counts each hit once. corpus.py's module docstring
    documents the same rule.
    """
    _write(tmp_path, "doc.md", _DOCUMENT)

    sections = {
        hit.section: hit.text for hit in read_sections(tmp_path / "doc.md", tmp_path)
    }

    assert "Child text." not in sections["parent"]


def test_section_order_follows_the_document(tmp_path: Path) -> None:
    """Order is part of the contract: two runs that list sections differently
    diff against each other for no reason."""
    _write(tmp_path, "doc.md", _DOCUMENT)

    anchors = [hit.section for hit in read_sections(tmp_path / "doc.md", tmp_path)]

    assert anchors == ["title", "parent", "child", "sibling"]


def test_doc_id_is_the_relative_path_without_the_suffix(tmp_path: Path) -> None:
    """`tutorial/dependencies/index.md` -> `tutorial/dependencies/index`.

    This is the exact string data/gold_questions.json stores, which is why it
    is tested on a nested path and not on a file in the root: a root-level file
    would pass even if relative_to or as_posix were missing.
    """
    _write(tmp_path, "tutorial/dependencies/index.md", "## Title {#title}\n")

    hits = list(read_sections(tmp_path / "tutorial/dependencies/index.md", tmp_path))

    assert hits[0].doc_id == "tutorial/dependencies/index"


def test_the_heading_title_is_part_of_the_section_text(tmp_path: Path) -> None:
    """The title carries the terms a search hits on. Dropping it costs recall
    on exactly the term_collision questions the gold set is built for."""
    _write(tmp_path, "doc.md", "## Dependency Injection {#di}\nSome content.\n")

    hits = list(read_sections(tmp_path / "doc.md", tmp_path))

    assert "Dependency Injection" in hits[0].text


def test_comments_in_a_code_block_are_not_headings(tmp_path: Path) -> None:
    """Inside ```python a `#` starts a comment. Without fence tracking every
    commented example opens a section, and most of them have no anchor, so the
    corpus would not even parse."""
    _write(
        tmp_path,
        "doc.md",
        "## Example {#example}\n```python\n# not a heading\n```\n",
    )

    hits = list(read_sections(tmp_path / "doc.md", tmp_path))

    assert len(hits) == 1
    assert "# not a heading" in hits[0].text


def test_a_heading_without_an_anchor_is_reported(tmp_path: Path) -> None:
    """A missing anchor is a hole in the corpus: nothing can cite that section.

    Python note: iter_sections is a generator, so the body does not run until
    something consumes it. `pytest.raises(CorpusError)` around the bare call
    would pass while nothing was ever parsed - hence the `list(...)`. `match`
    is a regex against the message, so the test pins what the error says, not
    only that something was raised.
    """
    _write(tmp_path, "doc.md", "## No Anchor\n")

    with pytest.raises(CorpusError, match="heading without anchor"):
        list(iter_sections(tmp_path))


def test_text_before_the_first_heading_is_reported(tmp_path: Path) -> None:
    """Text ahead of the first heading has no anchor. Dropping it quietly
    would hide corpus content that retrieval can return but no gold question
    can name - a permanent, invisible miss."""
    _write(tmp_path, "doc.md", "Stray text.\n## Title {#title}\n")

    with pytest.raises(CorpusError, match="before first heading"):
        list(iter_sections(tmp_path))


def test_the_source_note_is_not_part_of_the_corpus(tmp_path: Path) -> None:
    """SOURCE.md is provenance. It has unanchored headings, so leaving it in
    turns the whole corpus unparseable."""
    _write(tmp_path, "SOURCE.md", "## No Anchor\n")
    _write(tmp_path, "doc.md", "## Title {#title}\n")

    doc_ids = [hit.doc_id for hit in iter_sections(tmp_path)]

    assert doc_ids == ["doc"]


def test_the_vendored_corpus_reads() -> None:
    """Guards the real files, not a fixture: a corpus that stops parsing turns
    every expected_source into a dangling reference at once. The count is
    pinned - a vendor run that silently drops files is the failure this
    catches, and "more than zero" would not.
    """
    hits = list(iter_sections(CORPUS_ROOT))

    doc_ids = {hit.doc_id for hit in hits}
    assert len(doc_ids) == 24
    assert all(hit.text.strip() for hit in hits)
