"""Read the vendored corpus as sections.

A section, not a file, is the unit the gold questions point at. It survives
phase 3: a chunk maps back to (doc_id, section), while a chunk id would change
with every chunk size the harness varies, and an expected_source built on it
would be void after the first configuration change.
"""

import re
from collections.abc import Iterator
from pathlib import Path

from harness.core.interfaces import SectionHit

# `## Title { #anchor }`. The anchor is authored in the FastAPI docs, so it is
# read, never slugified: a slug derived here would drift from the anchor the
# docs actually publish, and both sides of the comparison would look fine.
#
# Python note: `(?P<name>...)` is a named group, read back as m.group("name").
# The `r"..."` prefix is a raw string, so `\s` stays `\s` and is not read as an
# escape sequence first. C# equivalent: @"..." plus (?<name>...).
HEADING_PATTERN = re.compile(
    r"^#{1,6}\s+(?P<title>.*?)\s*\{\s*#(?P<anchor>[\w-]+)\s*\}\s*$"
)
UNANCHORED_HEADING_PATTERN = re.compile(r"^#{1,6}\s+\S")
FENCE_PATTERN = re.compile(r"^```")

SOURCE_NOTE = "SOURCE.md"
"""Provenance, not corpus. It carries headings without anchors and no content
a question could ever cite."""


class CorpusError(Exception):
    """The corpus on disk is not usable as a source of gold answers."""


def iter_sections(corpus_root: Path) -> Iterator[SectionHit]:
    """Yield every section of every document under corpus_root.

    Sections do not nest. A section ends at the next heading of any level, so
    a `##` block does not carry the text of its `###` blocks a second time.
    Nesting them would count the same sentence once per level and inflate any
    later recall number without a single extra document being retrieved.

    Python note - `Iterator[T]` plus `yield`:
      C# equivalent: IEnumerable<T> with `yield return`. The body does not run
      when you call the function. It runs while something consumes it, one
      item per `next()`. Consequences you will hit in the tests:
        - `for hit in iter_sections(root)` consumes lazily.
        - `list(iter_sections(root))` forces the whole run.
        - A `raise` inside only fires once the consumer reaches that point,
          so `pytest.raises` needs `list(...)` around the call.
      `yield from other_generator()` forwards every item of the inner
      generator without a manual loop.
    """
    # sorted: the filesystem order is not guaranteed, and section order has to
    # be reproducible across machines, or two runs diff against each other.
    for path in sorted(corpus_root.rglob("*.md")):
        if path.name == SOURCE_NOTE:
            continue
        yield from read_sections(path, corpus_root)


def read_sections(path: Path, corpus_root: Path) -> Iterator[SectionHit]:
    """Split one file into sections.

    A small state machine over the lines. `section` holds the anchor of the
    section being collected, `body` its lines. `body = [...]` rebinds to a
    fresh list at every heading: `body.clear()` would empty the very list a
    just-yielded SectionHit still references.

    Returns:
        Sections in document order.

    Raises:
        CorpusError: On a heading without an anchor, or on text before the
            first heading.
    """
    doc_id = path.relative_to(corpus_root).with_suffix("").as_posix()
    section: str | None = None
    body: list[str] = []
    body_start_line: int | None = None
    in_fence = False

    # as_posix: without it a Windows run yields backslashes, and every doc_id
    # in data/gold_questions.json stops matching. Encoding is named because the
    # platform default is not utf-8 everywhere.
    text = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), 1):
        if FENCE_PATTERN.match(line):
            in_fence = not in_fence
            body.append(line)
            continue

        # Inside ```python a `#` starts a comment, not a section, and the
        # FastAPI docs are full of them.
        if in_fence:
            body.append(line)
            continue

        match = HEADING_PATTERN.match(line)
        if match is None:
            if UNANCHORED_HEADING_PATTERN.match(line):
                raise CorpusError(
                    f"{doc_id}:{line_number}: heading without anchor: {line!r}"
                )
            if body_start_line is None:
                body_start_line = line_number
            body.append(line)
            continue

        title = match.group("title")
        new_anchor = match.group("anchor")

        if section is not None:
            yield SectionHit(doc_id=doc_id, section=section, text=_join(body))
        elif _join(body):
            raise CorpusError(
                f"{doc_id}:{body_start_line}: "
                f"text before first heading: {_join(body)!r}"
            )

        section = new_anchor
        body = [title]

    # The file ends without a heading to close the last section. Forgetting
    # this drops exactly one section per file, and always the last one.
    if section is not None:
        yield SectionHit(doc_id=doc_id, section=section, text=_join(body))


def _join(body: list[str]) -> str:
    """Join the buffered lines into the section text.

    `"\\n".join(body)` is C# string.Join("\\n", body) with the separator first.
    `.strip()` removes the blank lines that a heading-to-heading slice always
    picks up at both ends, so an empty section is exactly the falsy empty
    string and the caller can test it with a plain `if`.
    """
    return "\n".join(body).strip()
