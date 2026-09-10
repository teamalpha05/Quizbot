"""HTML renderer for /google research reports."""

from __future__ import annotations

import html
import io
import re
from datetime import datetime, timezone
from typing import Any


def _esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _markdown_to_html(md: str) -> str:
    """Small dependency-free Markdown renderer for research reports."""
    text = md.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    in_ul = False
    in_ol = False

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def inline(s: str) -> str:
        s = _esc(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
        s = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"<em>\1</em>", s)
        s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', s)
        return s

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            close_lists()
            i += 1
            continue

        if line.startswith("### "):
            close_lists(); out.append(f"<h3>{inline(line[4:])}</h3>"); i += 1; continue
        if line.startswith("## "):
            close_lists(); out.append(f"<h2>{inline(line[3:])}</h2>"); i += 1; continue
        if line.startswith("# "):
            close_lists(); out.append(f"<h1>{inline(line[2:])}</h1>"); i += 1; continue

        if re.match(r"^[-*+] ", line):
            if not in_ul:
                close_lists(); out.append("<ul>"); in_ul = True
            out.append(f"<li>{inline(re.sub(r'^[-*+] ', '', line))}</li>")
            i += 1; continue

        if re.match(r"^\d+\. ", line):
            if not in_ol:
                close_lists(); out.append("<ol>"); in_ol = True
            out.append(f"<li>{inline(re.sub(r'^\d+\. ', '', line))}</li>")
            i += 1; continue

        # Markdown table.
        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            close_lists()
            table_lines = [line]
            i += 1
            while i < len(lines) and "|" in lines[i].strip() and lines[i].strip():
                table_lines.append(lines[i].strip()); i += 1
            rows = []
            for row in table_lines:
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                rows.append(cells)
            if len(rows) >= 2:
                out.append('<div class="table-scroll"><table>')
                out.append("<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr></thead>")
                out.append("<tbody>")
                for row in rows[2:]:
                    padded = row + [""] * max(0, len(rows[0]) - len(row))
                    out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in padded[:len(rows[0])]) + "</tr>")
                out.append("</tbody></table></div>")
            continue

        if line.startswith("> "):
            close_lists(); out.append(f'<blockquote>{inline(line[2:])}</blockquote>'); i += 1; continue

        # Paragraph: consume consecutive normal lines.
        para = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith(("# ", "## ", "### ", "> ")) or re.match(r"^[-*+] ", nxt) or re.match(r"^\d+\. ", nxt):
                break
            if "|" in nxt and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
                break
            para.append(nxt); i += 1
        close_lists()
        out.append(f"<p>{inline(' '.join(para))}</p>")

    close_lists()
    return "\n".join(out)


def build_google_report_html(query: str, article: str, results: list[dict[str, str]]) -> tuple[io.BytesIO, str]:
    body = _markdown_to_html(article)
    source_cards = []
    for i, item in enumerate(results, 1):
        source_cards.append(
            f'<a class="source" href="{_esc(item["link"])}" target="_blank" rel="noopener noreferrer">'
            f'<span class="source-no">{i}</span>'
            f'<span><strong>{_esc(item["title"])}</strong><small>{_esc(item["link"])}</small></span>'
            f'</a>'
        )

    stamp = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    title = _esc(query[:180])
    html_doc = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title} — Google Research</title>
<style>
:root {{
  --bg:#f4f6fb; --card:#fff; --text:#172033; --muted:#667085; --border:#dfe4ee;
  --primary:#2563eb; --primary-soft:#eff6ff; --table-head:#f8fafc;
}}
* {{ box-sizing:border-box; }}
html,body {{ margin:0; padding:0; background:var(--bg); color:var(--text); font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; line-height:1.65; }}
body {{ padding:18px 12px 40px; }}
.container {{ max-width:900px; margin:auto; }}
.hero {{ background:linear-gradient(135deg,#2563eb,#4f46e5); color:#fff; border-radius:18px; padding:22px 20px; box-shadow:0 12px 30px rgba(37,99,235,.18); margin-bottom:18px; }}
.hero .eyebrow {{ font-size:12px; font-weight:700; opacity:.85; text-transform:uppercase; letter-spacing:.08em; }}
.hero h1 {{ margin:5px 0 8px; font-size:25px; line-height:1.25; }}
.hero p {{ margin:0; opacity:.9; font-size:13px; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:16px; padding:20px; box-shadow:0 4px 16px rgba(15,23,42,.05); margin-bottom:18px; overflow:hidden; }}
h1,h2,h3 {{ line-height:1.3; }}
.card h1 {{ font-size:25px; margin:0 0 14px; }}
.card h2 {{ font-size:21px; margin:24px 0 10px; }}
.card h3 {{ font-size:17px; margin:20px 0 8px; }}
p {{ margin:10px 0; }}
strong {{ font-weight:750; }}
blockquote {{ margin:14px 0; padding:12px 14px; border-left:4px solid var(--primary); background:var(--primary-soft); border-radius:8px; }}
ul,ol {{ padding-left:24px; }}
li {{ margin:5px 0; }}
code {{ background:#eef2f7; border-radius:5px; padding:2px 5px; font-size:.92em; }}
a {{ color:var(--primary); }}
.table-scroll {{ width:100%; overflow-x:auto; overflow-y:hidden; -webkit-overflow-scrolling:touch; border:1px solid var(--border); border-radius:12px; margin:14px 0; }}
table {{ border-collapse:collapse; width:max-content; min-width:100%; background:#fff; }}
th,td {{ padding:11px 14px; border-right:1px solid var(--border); border-bottom:1px solid var(--border); text-align:left; vertical-align:top; white-space:normal; }}
th {{ background:var(--table-head); font-weight:700; }}
tr:last-child td {{ border-bottom:0; }}
th:last-child,td:last-child {{ border-right:0; }}
.sources {{ display:grid; gap:10px; }}
.source {{ display:flex; align-items:flex-start; gap:10px; text-decoration:none; color:inherit; padding:12px; border:1px solid var(--border); border-radius:12px; background:#fff; }}
.source:hover {{ background:#f8fafc; }}
.source-no {{ min-width:28px; height:28px; border-radius:50%; display:grid; place-items:center; background:var(--primary-soft); color:var(--primary); font-weight:800; font-size:12px; }}
.source strong {{ display:block; margin-bottom:2px; }}
.source small {{ color:var(--muted); display:block; overflow-wrap:anywhere; font-size:11px; line-height:1.4; }}
.footer {{ text-align:center; color:var(--muted); font-size:11px; padding:4px; }}
@media (max-width:600px) {{ body {{ padding:10px 8px 30px; }} .hero {{ padding:18px 16px; }} .hero h1 {{ font-size:21px; }} .card {{ padding:15px; border-radius:13px; }} .card h1 {{ font-size:22px; }} .card h2 {{ font-size:19px; }} th,td {{ padding:9px 11px; min-width:120px; }} }}
@media (prefers-color-scheme:dark) {{
  :root {{ --bg:#0f1320; --card:#161b2c; --text:#e7e9f5; --muted:#9aa2bd; --border:#2a3149; --table-head:#1e2438; --primary-soft:#1b2a4a; }}
  table {{ background:var(--card); }} .source {{ background:var(--card); }} .source:hover {{ background:#1e2438; }} code {{ background:#242b3d; }}
}}
</style>
</head>
<body>
<div class="container">
  <section class="hero">
    <div class="eyebrow">Google Research</div>
    <h1>{title}</h1>
    <p>Research report • Generated {stamp} • {len(results)} Google sources reviewed</p>
  </section>
  <main class="card">{body}</main>
  <section class="card">
    <h2 style="margin-top:0">Sources</h2>
    <div class="sources">{"".join(source_cards)}</div>
  </section>
  <div class="footer">Sources are provided for verification. Always open the original source for full context.</div>
</div>
</body>
</html>'''

    data = io.BytesIO(html_doc.encode("utf-8"))
    data.seek(0)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", query.strip())[:50].strip("-") or "google-research"
    return data, f"{safe_name}-research.html"
