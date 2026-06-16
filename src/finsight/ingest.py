from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field
from pathlib import Path

ITEM_PATTERN = re.compile(
    r"^\s*(?:ITEM|Item|I\s?T\s?E\s?M)\s+(\d{1,2}A?B?)\s*[.:–—-]?\s*(.{0,120})$",
    re.MULTILINE,
)
_TAG = re.compile(r"<[^>]+>")
_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_MIN_SECTION_CHARS = 200


@dataclass
class Section:
    item: str
    title: str
    text: str


@dataclass
class Document:
    ticker: str
    form: str
    date: str
    path: Path
    sections: list[Section] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return f"{self.ticker}_{self.form}_{self.date}"

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


def clean_text(raw: str) -> str:
    text = raw
    if "</" in text[:5000] or "<html" in text[:2000].lower():
        text = _SCRIPT_STYLE.sub(" ", text)
        text = re.sub(r"</?(p|div|tr|table|h\d|li|br)[^>]*>", "\n\n", text, flags=re.IGNORECASE)
        text = _TAG.sub(" ", text)
        text = html_lib.unescape(text)
    text = text.replace(" ", " ").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sections(text: str) -> list[Section]:
    matches = list(ITEM_PATTERN.finditer(text))
    if not matches:
        return [Section(item="0", title="Full document", text=text.strip())]

    best: dict[str, Section] = {}
    order: list[str] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) < _MIN_SECTION_CHARS:
            continue
        item = m.group(1).upper()
        if item not in best or len(body) > len(best[item].text):
            if item not in best:
                order.append(item)
            best[item] = Section(item=item, title=m.group(2).strip(" .:–—-"), text=body)

    sections = [best[i] for i in order]
    return sections or [Section(item="0", title="Full document", text=text.strip())]


def load_document(path: Path) -> Document:
    name = path.stem
    parts = name.split("_")
    if len(parts) != 3:
        raise ValueError(f"Bad filename (want TICKER_FORM_DATE.txt): {path.name}")
    ticker, form, date = parts
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    return Document(ticker=ticker, form=form, date=date, path=path,
                    sections=_split_sections(text))


def load_corpus(*dirs: Path) -> list[Document]:
    docs: list[Document] = []
    for d in dirs:
        if not d.exists():
            continue
        for pattern in ("*.txt", "*.htm", "*.html"):
            for p in sorted(d.glob(pattern)):
                try:
                    docs.append(load_document(p))
                except ValueError:
                    continue
    return docs
