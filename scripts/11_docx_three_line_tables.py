"""
Apply the three-line table format required by 『재무관리연구』 to a pandoc-generated docx,
center header rows and columns whose body cells are all numeric (text columns stay left-aligned),
fix every table to the full text width (13.0 cm) with content-proportional columns, and center the bold <표 n>/[그림 n] title lines.

KFMA rule (원고작성 방법 9항): rows are separated only where necessary, with a single underline.
Each table gets a thick rule at the top and bottom and a thin rule under the header row;
no vertical lines, no rules between body rows, no shading. Borders are written directly
on each table and header cell (not only in the table style) so that Word, HWP import,
Google Docs, and macOS Quick Look all render them.

Usage:
    python scripts/11_docx_three_line_tables.py manuscript/manuscript_full_ko.docx [more.docx ...]
"""

from __future__ import annotations

import html
import re
import shutil
import sys
import tempfile
import zipfile

THICK, THIN = 12, 6  # eighths of a point: 1.5pt, 0.75pt
TEXT_WIDTH = 7370  # twips: 16.8 cm page - 2 x 1.9 cm margins = 13.0 cm (KFMA Word layout)

TBL_BORDERS = (
    "<w:tblBorders>"
    f'<w:top w:val="single" w:sz="{THICK}" w:space="0" w:color="000000"/>'
    '<w:left w:val="nil"/>'
    f'<w:bottom w:val="single" w:sz="{THICK}" w:space="0" w:color="000000"/>'
    '<w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/>'
    "</w:tblBorders>"
)
HEADER_TC_BORDERS = (
    "<w:tcBorders>"
    f'<w:top w:val="single" w:sz="{THICK}" w:space="0" w:color="000000"/>'
    f'<w:bottom w:val="single" w:sz="{THIN}" w:space="0" w:color="000000"/>'
    "</w:tcBorders>"
)


def fix_table(tbl: str) -> str:
    # table-level borders
    tbl = re.sub(r"<w:tblBorders>.*?</w:tblBorders>", "", tbl, flags=re.S)
    # every table spans the full text width (100% = 5000 fiftieths of a percent)
    tbl = re.sub(r"<w:tblW [^>]*/>", f'<w:tblW w:type="dxa" w:w="{TEXT_WIDTH}"/>' + TBL_BORDERS
                 + '<w:tblLayout w:type="fixed"/>', tbl, count=1)
    assert TBL_BORDERS in tbl, "tblW not found"
    # header row = first <w:tr>
    m = re.search(r"<w:tr>.*?</w:tr>", tbl, flags=re.S)
    row = m.group(0)
    row = re.sub(r"<w:tcBorders>.*?</w:tcBorders>", "", row, flags=re.S)
    row = re.sub(r"<w:tcPr\s*/>", "<w:tcPr></w:tcPr>", row)
    row = row.replace("<w:tcPr>", "<w:tcPr>" + HEADER_TC_BORDERS)
    # bold header text (only where a run is not already bold)
    row = re.sub(r"<w:r>(?!<w:rPr>)", "<w:r><w:rPr></w:rPr>", row)

    def bold(rp: re.Match) -> str:
        inner = rp.group(1)
        return rp.group(0) if re.search(r"<w:b\s*/>", inner) else "<w:rPr><w:b/>" + inner + "</w:rPr>"

    row = re.sub(r"<w:rPr>(.*?)</w:rPr>", bold, row, flags=re.S)
    # header row: every column title centered
    row = re.sub(r"<w:tc>.*?</w:tc>", lambda c: set_jc(c.group(0), "center"), row, flags=re.S)
    return tbl[: m.start()] + row + tbl[m.end():]


TITLE = re.compile(r"^(<표 [A-Z]?\d+>|\[그림 \d+\]|<Table [A-Z]?\d+>|\[Figure \d+\]) ")


def center_titles(doc: str) -> tuple[str, int]:
    """Center bold table/figure title paragraphs (e.g. '<표 5> 전체 기술통계') and drop their first-line indent."""
    n = 0

    def para(p: re.Match) -> str:
        nonlocal n
        s = p.group(0)
        if "<w:tbl" in s:
            return s
        text = html.unescape("".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", s)))
        first_run = re.search(r"<w:r>.*?</w:r>", s, flags=re.S)
        if not TITLE.match(text) or not first_run or not re.search(r"<w:b\s*/>", first_run.group(0)):
            return s
        n += 1
        s = re.sub(r"<w:jc [^>]*/>|<w:ind [^>]*/>", "", s)
        props = '<w:ind w:firstLine="0"/><w:jc w:val="center"/>'
        if re.search(r"<w:pPr\s*/>", s):
            return re.sub(r"<w:pPr\s*/>", f"<w:pPr>{props}</w:pPr>", s, count=1)
        if "<w:pPr>" in s:
            return s.replace("</w:pPr>", props + "</w:pPr>", 1)
        return s.replace("<w:p>", f"<w:p><w:pPr>{props}</w:pPr>", 1)

    return re.sub(r"<w:p>(?:(?!<w:p>).)*?</w:p>", para, doc, flags=re.S), n


NUMERIC = re.compile(r"^[\s\d.,%+\-−–—<>=≥≤/\[\]()~×n]*$")


def cell_text(tc: str) -> str:
    raw = "".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", tc)).strip()
    return html.unescape(raw)


def is_numeric(text: str) -> bool:
    return bool(text) and NUMERIC.match(text) is not None and (re.search(r"\d", text) or text in {"—", "---", "–"})


def set_jc(tc: str, val: str) -> str:
    def para(p: re.Match) -> str:
        s = p.group(0)
        s = re.sub(r"<w:jc [^>]*/>", "", s)
        if re.search(r"<w:pPr\s*/>", s):
            return re.sub(r"<w:pPr\s*/>", f'<w:pPr><w:jc w:val="{val}"/></w:pPr>', s, count=1)
        if "<w:pPr>" in s:
            return s.replace("</w:pPr>", f'<w:jc w:val="{val}"/></w:pPr>', 1)
        return s.replace("<w:p>", f'<w:p><w:pPr><w:jc w:val="{val}"/></w:pPr>', 1)

    return re.sub(r"<w:p>.*?</w:p>", para, tc, flags=re.S)


def center_numeric_columns(tbl: str) -> str:
    """Center every column (header included) whose body cells are all numeric; leave text columns left."""
    rows = list(re.finditer(r"<w:tr>.*?</w:tr>", tbl, flags=re.S))
    if len(rows) < 2:
        return tbl
    cells = [re.findall(r"<w:tc>.*?</w:tc>", r.group(0), flags=re.S) for r in rows]
    ncol = len(cells[0])
    numeric_cols = {
        j for j in range(ncol)
        if all(j < len(r) and is_numeric(cell_text(r[j])) for r in cells[1:])
    }
    if not numeric_cols:
        return tbl
    out, pos = [], 0
    for r, rc in zip(rows, cells):
        row = r.group(0)
        for j, tc in enumerate(rc):
            if j in numeric_cols:
                row = row.replace(tc, set_jc(tc, "center"), 1)
        out.append(tbl[pos:r.start()] + row)
        pos = r.end()
    return "".join(out) + tbl[pos:]


def display_len(text: str) -> int:
    """Approximate printed width: CJK characters count double."""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


CODE = re.compile(r"^[A-Za-z0-9_./:-]+$")
CODE_SZ = 16  # half-points: 8pt for code-name columns (e.g. variable names)
BODY_PT, CODE_PT = 10, 8


def code_columns(cells: list[list[str]]) -> set[int]:
    """Columns (other than the first) whose body cells are all code identifiers such as riskiness_score."""
    if len(cells) < 2:
        return set()
    ncol = max(len(r) for r in cells)
    cols = set()
    for j in range(1, ncol):
        body = [cell_text(r[j]) for r in cells[1:] if j < len(r)]
        if body and all(CODE.match(t) for t in body) and any("_" in t for t in body):
            cols.add(j)
    return cols


def shrink_code_columns(tbl: str) -> str:
    """Set body cells of code-name columns to 8pt (font unchanged)."""
    rows = list(re.finditer(r"<w:tr>.*?</w:tr>", tbl, flags=re.S))
    cells = [re.findall(r"<w:tc>.*?</w:tc>", r.group(0), flags=re.S) for r in rows]
    cols = code_columns(cells)
    if not cols:
        return tbl
    size = f'<w:sz w:val="{CODE_SZ}"/><w:szCs w:val="{CODE_SZ}"/>'

    def small(tc: str) -> str:
        tc = re.sub(r"<w:r>(?!<w:rPr>)", "<w:r><w:rPr></w:rPr>", tc)
        tc = re.sub(r"<w:sz [^>]*/>|<w:szCs [^>]*/>", "", tc)
        return re.sub(r"(<w:r><w:rPr>.*?)</w:rPr>", lambda m: m.group(1) + size + "</w:rPr>", tc, flags=re.S)

    out, pos = [], 0
    for i, (r, rc) in enumerate(zip(rows, cells)):
        row = r.group(0)
        if i > 0:
            for j, tc in enumerate(rc):
                if j in cols:
                    row = row.replace(tc, small(tc), 1)
        out.append(tbl[pos:r.start()] + row)
        pos = r.end()
    return "".join(out) + tbl[pos:]


def fit_width(tbl: str) -> str:
    """Fix the table to the full text width and split it among columns in proportion to content length."""
    rows = re.findall(r"<w:tr>.*?</w:tr>", tbl, flags=re.S)
    cells = [re.findall(r"<w:tc>.*?</w:tc>", r, flags=re.S) for r in rows]
    ncol = max(len(r) for r in cells)
    PAD, UNIT = 3, 92  # padding in characters; twips per half-width character (approx. 9pt font)
    minimum, ideal = [], []
    code_cols = code_columns(cells)
    for j in range(ncol):
        texts = [cell_text(r[j]) for r in cells if j < len(r)]
        scale = CODE_PT / BODY_PT if j in code_cols else 1.0
        longest_word = max([display_len(texts[0].split()[0]) if texts and texts[0].split() else 0]
                           + [round(display_len(w) * scale) for t in texts[1:] for w in t.split()])
        longest_body = max((round(display_len(t) * scale) for t in texts[1:]), default=0)
        minimum.append(longest_word + PAD)              # never break inside a word
        header = display_len(texts[0]) if texts else 0
        ideal.append(max(min(longest_body, 60), min(header, 16), longest_word) + PAD)
    numeric = [j for j in range(ncol)
               if len(cells) > 1 and all(j < len(r) and is_numeric(cell_text(r[j])) for r in cells[1:])]
    base = [m * UNIT for m in minimum]
    goal = [i * UNIT for i in ideal]
    if sum(base) >= TEXT_WIDTH:                          # too many columns: scale minimums down
        widths = [TEXT_WIDTH * b / sum(base) for b in base]
    elif sum(goal) >= TEXT_WIDTH:                        # grow toward ideal widths as far as room allows
        deficit = [g - b for g, b in zip(goal, base)]
        room = TEXT_WIDTH - sum(base)
        widths = [b + room * d / sum(deficit) for b, d in zip(base, deficit)]
    else:                                                # all ideals fit: share leftover evenly
        left = TEXT_WIDTH - sum(goal)
        share = numeric or list(range(ncol))             # numeric tables widen number columns equally
        widths = [g + (left / len(share) if j in share else 0) for j, g in enumerate(goal)]
    widths = [round(w) for w in widths]
    widths[-1] += TEXT_WIDTH - sum(widths)
    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>"
    tbl = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", grid, tbl, count=1, flags=re.S)

    def row_fix(r: re.Match) -> str:
        row = r.group(0)
        out, pos = [], 0
        for j, c in enumerate(re.finditer(r"<w:tc>.*?</w:tc>", row, flags=re.S)):
            tc = re.sub(r"<w:tcW [^>]*/>", "", c.group(0))
            tcw = f'<w:tcW w:w="{widths[min(j, ncol - 1)]}" w:type="dxa"/>'
            tc = re.sub(r"<w:tcPr\s*/>", "<w:tcPr></w:tcPr>", tc, count=1)
            tc = tc.replace("<w:tcPr>", "<w:tcPr>" + tcw, 1) if "<w:tcPr>" in tc else tc.replace("<w:tc>", f"<w:tc><w:tcPr>{tcw}</w:tcPr>", 1)
            out.append(row[pos:c.start()] + tc)
            pos = c.end()
        return "".join(out) + row[pos:]

    return re.sub(r"<w:tr>.*?</w:tr>", row_fix, tbl, flags=re.S)


def process(path: str) -> int:
    with zipfile.ZipFile(path) as zin:
        doc = zin.read("word/document.xml").decode("utf-8")
        n = 0

        def repl(m: re.Match) -> str:
            nonlocal n
            n += 1
            return fit_width(shrink_code_columns(center_numeric_columns(fix_table(m.group(0)))))

        doc = re.sub(r"<w:tbl>.*?</w:tbl>", repl, doc, flags=re.S)
        doc, n_titles = center_titles(doc)
        print(f"  centered titles: {n_titles}")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for it in zin.infolist():
                data = doc.encode("utf-8") if it.filename == "word/document.xml" else zin.read(it.filename)
                zout.writestr(it, data)
    shutil.move(tmp, path)
    return n


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, "tables:", process(p))
