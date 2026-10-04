"""Node A: synthetic report and QA generation.

Generates dummy cross-department project retrospectives for the internal knowledge-search RAG demo.

- No company names, real or fictional: an anonymous set of departments within one company.
- Each department uses slightly different heading vocabulary, mimicking inconsistent house styles.
- File format (Markdown/Word/Excel/PowerPoint/PDF) is random per report, not tied to a department.

Uses a local OpenAI-compatible LLM (e.g. LM Studio). Run with `python -m silo_rag.datagen`.
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .config import EVAL_DIR, SYNTH_REPORTS_DIR, load_config
from .llm_client import LLMClient, LLMConnectionError

# --- Domain vocabulary ------------------------------------------------------

PROJECT_TYPES = [
    "新規事業立ち上げ",
    "業務プロセス改善",
    "新システム導入",
    "マーケティング施策",
    "組織改編・体制変更",
]

METHODS = ["アジャイル（スクラム）", "ウォーターフォール", "OKR運用"]

DEPARTMENTS: dict[str, list[str]] = {
    "マーケティング部": [
        "新商品ローンチキャンペーン", "ブランドリニューアル", "SNS運用強化",
        "展示会出展", "顧客アンケート改善",
    ],
    "営業推進部": [
        "新規開拓施策", "既存顧客深耕プログラム", "代理店連携強化",
        "価格改定対応", "商談プロセス標準化",
    ],
    "商品開発部": [
        "新商品企画", "既存商品リニューアル", "試作評価プロセス改善",
        "サプライヤー切り替え", "コスト削減プロジェクト",
    ],
    "カスタマーサポート部": [
        "問い合わせ対応フロー刷新", "FAQサイト刷新", "サポート体制見直し",
        "クレーム対応プロセス改善", "顧客満足度調査",
    ],
    "経営企画部": [
        "中期計画策定", "新拠点立ち上げ", "組織統合プロセス",
        "予算策定プロセス見直し", "組織改編",
    ],
}

# Per-department heading vocabulary, mimicking terminology that isn't unified across departments.
DEPT_TERMINOLOGY: dict[str, dict[str, str]] = {
    "マーケティング部": {"background": "実施条件", "approach": "推進体制"},
    "営業推進部": {"background": "前提条件", "approach": "実行体制"},
    "商品開発部": {"background": "制約条件", "approach": "進め方"},
    "カスタマーサポート部": {"background": "実行条件", "approach": "対応フロー"},
    "経営企画部": {"background": "与件", "approach": "実行計画"},
}

# Assigned randomly per report, not per department; departmental inconsistency lives in
# DEPT_TERMINOLOGY instead.
FILE_FORMATS = ["md", "docx", "xlsx", "pdf", "pptx"]

# Fixed metadata field order shared by every writer.
METADATA_FIELDS = ["report_id", "dept", "project_type", "subject", "method", "resourcing", "author", "date"]

LESSONS = [
    "関係部署への事前説明が不十分で、後工程で手戻りが発生した",
    "進捗の可視化が不十分で、問題の発覚が遅れた",
    "現場の意見を十分に吸い上げないまま計画を進め、実行段階で抵抗にあった",
    "KPIの設定が曖昧で、成果の評価基準が途中でぶれた",
    "外部ベンダーとの役割分担が不明確で、責任の所在が曖昧になった",
    "過去の類似施策の教訓を参照せず、同じ失敗を繰り返した",
    "意思決定のスピードが遅く、市場機会を逃した",
    "担当者の異動により、ノウハウが引き継がれず停滞した",
    "予算超過に気づくのが遅れ、途中でスコープを縮小せざるを得なかった",
    "成功体験を横展開する仕組みがなく、他部署では再現されなかった",
]

RESOURCING = [
    "既存メンバーのみで対応", "外部コンサルティング活用", "他部署からの兼任メンバーで構成",
    "新規採用メンバー中心", "外部ベンダー・委託中心",
]

AUTHOR_NAMES = [f"担当者{c}" for c in "ABCDEFGHIJ"]


@dataclass
class ReportSpec:
    report_id: str
    dept: str
    project_type: str
    subject: str
    method: str
    resourcing: str
    lesson: str
    author: str
    report_date: date
    file_format: str


def generate_report_specs(count: int, *, seed: int | None = None) -> list[ReportSpec]:
    """Build report specs that cover every department/project-type pair (stratified sampling)."""
    rng = random.Random(seed)
    depts = list(DEPARTMENTS.keys())
    combos = [(d, a) for d in depts for a in PROJECT_TYPES]
    rng.shuffle(combos)

    specs: list[ReportSpec] = []
    base_date = date(2023, 1, 1)
    for i in range(count):
        dept, project_type = combos[i % len(combos)]
        subject = rng.choice(DEPARTMENTS[dept])
        specs.append(
            ReportSpec(
                report_id=f"RPT-{i + 1:03d}",
                dept=dept,
                project_type=project_type,
                subject=subject,
                method=rng.choice(METHODS),
                resourcing=rng.choice(RESOURCING),
                lesson=rng.choice(LESSONS),
                author=rng.choice(AUTHOR_NAMES),
                report_date=base_date + timedelta(days=rng.randint(0, 900)),
                file_format=rng.choice(FILE_FORMATS),
            )
        )
    return specs


def build_prompt(spec: ReportSpec) -> tuple[str, str]:
    term = DEPT_TERMINOLOGY[spec.dept]
    system = (
        "あなたは事業会社の企画・推進担当者です。"
        "実在・架空を問わず企業名や具体的なブランド名は一切書かないでください。"
        "社内向けのプロジェクト振り返りレポートをMarkdown形式で日本語で書いてください。"
        "数値は具体的な架空の値を使ってよいですが、実在企業の公表値を模倣しないでください。"
    )
    user = f"""以下の条件で、社内のプロジェクト振り返りレポート本文をMarkdown形式で書いてください。

- 作成部署: {spec.dept}
- 対象テーマ: {spec.subject}
- プロジェクト種別: {spec.project_type}
- 採用手法: {spec.method}
- 主要リソース: {spec.resourcing}
- 担当者: {spec.author}
- 日付: {spec.report_date.isoformat()}
- 今回のプロジェクトで実際に起きた教訓・つまずいたポイント: {spec.lesson}

以下の見出し構成に厳密に従ってください（各見出しは `## ` で始める）:

## プロジェクト目的
## 対象領域・テーマ
## {term["background"]}
## {term["approach"]}
## 主要リソース
## 採用手法
## 成果サマリー
## 教訓・つまずいたポイント

「教訓・つまずいたポイント」セクションには、
上記の「実際に起きた教訓・つまずいたポイント」を具体的な数値・状況付きで詳しく書いてください。
各セクションは3〜6行程度の具体的な記述にしてください。
タイトル行（# で始まる見出し）は書かず、上記の `## ` 見出しから書き始めてください。"""
    return system, user


def _metadata_dict(spec: ReportSpec) -> dict[str, str]:
    return {
        "report_id": spec.report_id,
        "dept": spec.dept,
        "project_type": spec.project_type,
        "subject": spec.subject,
        "method": spec.method,
        "resourcing": spec.resourcing,
        "author": spec.author,
        "date": spec.report_date.isoformat(),
    }


_SECTION_HEADING_RE = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split the body into (heading, text) pairs at each `## heading`.

    Mirrors ingest.split_into_sections() but is duplicated here so datagen doesn't depend on
    ingest (which would couple against the DAG direction).
    """
    matches = list(_SECTION_HEADING_RE.finditer(body))
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append((heading, body[start:end].strip()))
    return sections


def _validate_report_body(spec: ReportSpec, body: str) -> None:
    """Check that the generated body has the `## heading` structure ingest expects.

    A wrong heading level or wording, or an empty section, would silently drop content from the
    index, so reject it here before it can overwrite the previous good dataset.
    """
    term = DEPT_TERMINOLOGY[spec.dept]
    required = [
        "プロジェクト目的",
        "対象領域・テーマ",
        term["background"],
        term["approach"],
        "主要リソース",
        "採用手法",
        "成果サマリー",
        "教訓・つまずいたポイント",
    ]
    matches = list(_SECTION_HEADING_RE.finditer(body))
    headings = [m.group(1).strip() for m in matches]

    missing = [h for h in required if h not in headings]
    if missing:
        raise LLMConnectionError(f"{spec.report_id}: 生成レポートに必須セクションが欠けています: {missing}")

    # Duplicate headings would produce colliding chunk IDs and make the Chroma insert fail.
    duplicated = sorted({h for h in headings if headings.count(h) > 1})
    if duplicated:
        raise LLMConnectionError(f"{spec.report_id}: 見出しが重複しています: {duplicated}")

    ordered = list(zip(headings, matches, strict=True))
    for idx, (heading, m) in enumerate(ordered):
        if heading not in required:
            continue
        start = m.end()
        end = ordered[idx + 1][1].start() if idx + 1 < len(ordered) else len(body)
        if not body[start:end].strip():
            raise LLMConnectionError(f"{spec.report_id}: セクション「{heading}」の本文が空です。")


_MAX_GENERATION_ATTEMPTS = 3

# A report can still break the heading structure after all per-report retries; rather than make the
# user rerun by hand, restart the whole batch (with temperature sampling, a rerun usually succeeds).
_MAX_BATCH_ATTEMPTS = 3

# Section that always gets the result image. Safe to hard-code: only background/approach vary
# by department.
RESULT_IMAGE_SECTION = "成果サマリー"


def _generate_result_image(spec: ReportSpec) -> bytes:
    """Draw a plausible dummy result chart for the project type with matplotlib.

    Not real data; it only exists so reports contain an embedded image. Seeded from report_id.
    """
    import io

    import matplotlib

    matplotlib.use("Agg")  # headless: no GUI backend
    import matplotlib.pyplot as plt
    import numpy as np

    # The default DejaVu Sans has no Japanese glyphs (tofu boxes plus a UserWarning), so prefer
    # common OS Japanese fonts and fall back to DejaVu Sans only if none is installed.
    plt.rcParams["font.sans-serif"] = [
        "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo",
        "Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans",
    ]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    rng = np.random.default_rng(abs(hash(spec.report_id)) % (2**32))
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)

    if spec.project_type == "マーケティング施策":
        t = np.linspace(0, 8, 40)
        kpi = 2.0 + np.where(t > 4, (t - 4) * 0.6, 0) + rng.normal(0, 0.15, t.shape)
        ax.plot(t, kpi)
        ax.axvline(4, linestyle="--", color="gray")
        ax.set_xlabel("週")
        ax.set_ylabel("コンバージョン率 [%]")
        ax.set_title(f"{spec.subject} KPI推移")
    elif spec.project_type == "新システム導入":
        t = np.linspace(0, 12, 40)
        adoption = 100 / (1 + np.exp(-(t - 6))) + rng.normal(0, 2, t.shape)
        ax.plot(t, adoption)
        ax.set_xlabel("月")
        ax.set_ylabel("利用率 [%]")
        ax.set_title(f"{spec.subject} 利用率推移")
    else:
        labels = ["施策前", "施策後"]
        before = rng.uniform(50, 80)
        after = before + rng.uniform(5, 25)
        ax.bar(labels, [before, after], color=["#888888", "#4c72b0"])
        ax.set_ylabel("主要KPI")
        ax.set_title(f"{spec.subject} 成果比較")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _generate_one_report(
    client: LLMClient, spec: ReportSpec
) -> tuple[dict[str, str], list[tuple[str, str]], bytes]:
    """Generate one report in memory; return (metadata, [(heading, text), ...], result PNG bytes).

    Small local LLMs don't always follow the heading structure (1 in 60 reports once failed on a
    missing heading), so retry a few times; with temperature 0.7 each attempt differs.
    Nothing is written here: generate_reports() writes only after every report succeeds.
    """
    system, user = build_prompt(spec)
    last_error: LLMConnectionError | None = None
    for attempt in range(1, _MAX_GENERATION_ATTEMPTS + 1):
        try:
            body = client.chat(system, user, temperature=0.7)
            _validate_report_body(spec, body)
        except LLMConnectionError as exc:
            last_error = exc
            print(f"{spec.report_id}: 生成/検証に失敗（{attempt}/{_MAX_GENERATION_ATTEMPTS}回目）: {exc}")
            continue
        image = _generate_result_image(spec)
        return _metadata_dict(spec), _split_sections(body), image

    assert last_error is not None
    raise last_error


# --- Per-format writers -----------------------------------------------------


def _report_title(metadata: dict[str, str]) -> str:
    return f"{metadata['subject']} {metadata['project_type']} 振り返りレポート（{metadata['report_id']}）"


def _write_markdown(
    metadata: dict[str, str], sections: list[tuple[str, str]], image: bytes, path: Path
) -> None:
    image_path = path.with_suffix(".png")
    image_path.write_bytes(image)

    lines = ["---", *(f"{key}: {metadata[key]}" for key in METADATA_FIELDS), "---", ""]
    header = "\n".join(lines)
    title = f"# {_report_title(metadata)}\n\n"
    parts = []
    for heading, text in sections:
        if heading == RESULT_IMAGE_SECTION:
            text = f"{text}\n\n![成果画像]({image_path.name})"
        parts.append(f"## {heading}\n{text}")
    body = "\n\n".join(parts) + "\n"
    path.write_text(header + title + body, encoding="utf-8")


def _write_docx(metadata: dict[str, str], sections: list[tuple[str, str]], image: bytes, path: Path) -> None:
    import io

    from docx import Document
    from docx.shared import Inches

    doc = Document()
    doc.add_heading(_report_title(metadata), level=1)

    table = doc.add_table(rows=len(METADATA_FIELDS), cols=2)
    for row, key in zip(table.rows, METADATA_FIELDS, strict=True):
        row.cells[0].text = key
        row.cells[1].text = metadata[key]

    for heading, text in sections:
        doc.add_heading(heading, level=2)
        doc.add_paragraph(text)
        if heading == RESULT_IMAGE_SECTION:
            doc.add_picture(io.BytesIO(image), width=Inches(4))

    doc.save(str(path))


def _write_xlsx(metadata: dict[str, str], sections: list[tuple[str, str]], image: bytes, path: Path) -> None:
    import io

    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage

    wb = Workbook()
    ws = wb.active
    ws.title = "report"
    for i, key in enumerate(METADATA_FIELDS, start=1):
        ws.cell(row=i, column=1, value=key)
        ws.cell(row=i, column=2, value=metadata[key])

    start_row = len(METADATA_FIELDS) + 2  # one blank row after the metadata
    image_anchor_row = start_row
    for offset, (heading, text) in enumerate(sections):
        row = start_row + offset
        ws.cell(row=row, column=1, value=heading)
        ws.cell(row=row, column=2, value=text)
        if heading == RESULT_IMAGE_SECTION:
            image_anchor_row = row

    # Columns C onward are empty, so the image won't cover any text.
    ws.add_image(XLImage(io.BytesIO(image)), f"D{image_anchor_row}")

    wb.save(str(path))


def _write_pptx(metadata: dict[str, str], sections: list[tuple[str, str]], image: bytes, path: Path) -> None:
    import io

    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    layout = prs.slide_layouts[1]  # Title and Content

    meta_slide = prs.slides.add_slide(layout)
    meta_slide.shapes.title.text = "レポートメタデータ"
    meta_slide.placeholders[1].text_frame.text = "\n".join(
        f"{key}: {metadata[key]}" for key in METADATA_FIELDS
    )

    for heading, text in sections:
        slide = prs.slides.add_slide(layout)
        slide.shapes.title.text = heading
        slide.placeholders[1].text_frame.text = text
        if heading == RESULT_IMAGE_SECTION:
            slide.shapes.add_picture(io.BytesIO(image), Inches(5.2), Inches(1.6), width=Inches(4))

    prs.save(str(path))


def _pdf_lines(metadata: dict[str, str], sections: list[tuple[str, str]]) -> list[str]:
    """Build the text lines drawn on the PDF pages (pure function, no reportlab).

    ingest._parse_pdf_text() relies on this line layout. Kept separate from drawing so the
    write/read round trip can be tested without reportlab or pypdf installed.
    """
    import textwrap

    # reportlab's drawString doesn't wrap, so pre-wrap at about 38 full-width characters.
    lines: list[str] = [f"{key}: {metadata[key]}" for key in METADATA_FIELDS]
    lines.append("")
    for heading, text in sections:
        lines.append(f"## {heading}")
        for raw_line in text.splitlines() or [""]:
            lines.extend(textwrap.wrap(raw_line, width=38) or [""])
        lines.append("")
    return lines


def _write_pdf(metadata: dict[str, str], sections: list[tuple[str, str]], image: bytes, path: Path) -> None:
    import io

    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    font_name = "HeiseiKakuGo-W5"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))

    lines = _pdf_lines(metadata, sections)
    image_reader = ImageReader(io.BytesIO(image))
    img_width, img_height = 200, 140

    page_width, page_height = 595, 842  # A4 (pt)
    left_margin, top_margin, bottom_margin = 40, 800, 40
    font_size, line_height = 11, 16

    c = canvas.Canvas(str(path), pagesize=(page_width, page_height))
    y = top_margin
    c.setFont(font_name, font_size)

    def _draw_image() -> None:
        nonlocal y
        if y - img_height < bottom_margin:
            c.showPage()
            c.setFont(font_name, font_size)
            y = top_margin
        c.drawImage(image_reader, left_margin, y - img_height, width=img_width, height=img_height)
        y -= img_height + line_height

    # Draw the image at the end of the result section: before the next heading or after the last line.
    in_result_section = False
    for line in lines:
        if line.startswith("## "):
            if in_result_section:
                _draw_image()
            in_result_section = line[3:].strip() == RESULT_IMAGE_SECTION

        if y < bottom_margin:
            c.showPage()
            c.setFont(font_name, font_size)
            y = top_margin
        c.drawString(left_margin, y, line)
        y -= line_height

    if in_result_section:
        _draw_image()

    c.showPage()
    c.save()


_WRITERS = {
    "md": _write_markdown,
    "docx": _write_docx,
    "xlsx": _write_xlsx,
    "pptx": _write_pptx,
    "pdf": _write_pdf,
}


def generate_reports(client: LLMClient, specs: list[ReportSpec], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate everything before writing, so a mid-run failure can't break the previous dataset
    # (or its matching qa_pairs.json).
    generated: dict[str, tuple[dict[str, str], list[tuple[str, str]], bytes]] = {}
    for spec in specs:
        generated[spec.report_id] = _generate_one_report(client, spec)
        print(f"generated (in-memory): {spec.report_id}")

    # Only now clear the previous output (every format plus Markdown's PNGs), so rerunning with a
    # smaller --count doesn't leave stale reports that contradict the new QA pairs.
    for fmt in [*FILE_FORMATS, "png"]:
        for stale in out_dir.glob(f"RPT-*.{fmt}"):
            stale.unlink()
    for spec in specs:
        metadata, sections, image = generated[spec.report_id]
        path = out_dir / f"{spec.report_id}.{spec.file_format}"
        _WRITERS[spec.file_format](metadata, sections, image, path)
        print(f"wrote: {path}")


# --- Gold-standard evaluation QA pairs --------------------------------------


def generate_eval_qa(specs: list[ReportSpec], count: int, *, seed: int | None = None) -> list[dict]:
    """Build evaluation QA pairs, some asked from a department unaware of another's past cases.

    Built mechanically from the specs, with no LLM call, so the gold standard stays easy to verify by hand.
    """
    rng = random.Random(seed)
    chosen = rng.sample(specs, k=min(count, len(specs)))

    # Questions only mention subject and project type, so every report sharing that pair is a valid
    # answer. The gold set lists all of them, each with its own dept/evidence.
    reports_by_key: dict[tuple[str, str], list[str]] = {}
    spec_by_report_id: dict[str, ReportSpec] = {}
    for s in specs:
        reports_by_key.setdefault((s.subject, s.project_type), []).append(s.report_id)
        spec_by_report_id[s.report_id] = s

    qa_pairs: list[dict] = []
    for i, spec in enumerate(chosen):
        cross_dept = i % 3 == 0  # every third question is cross-department
        matching_ids = reports_by_key[(spec.subject, spec.project_type)]
        if cross_dept:
            other_depts = [d for d in DEPARTMENTS if d != spec.dept]
            asking_dept = rng.choice(other_depts)
            question = (
                f"{asking_dept}です。{spec.subject}に似たテーマで{spec.project_type}を検討しています。"
                f"他部署で参考になりそうな過去の{spec.project_type}の事例はありますか？"
                "特に気をつけるべき落とし穴があれば教えてください。"
            )
            # The question asks for other departments' cases, so exclude the asker's own reports.
            gold_ids = [r for r in matching_ids if spec_by_report_id[r].dept != asking_dept]
        else:
            asking_dept = spec.dept
            question = (
                f"{spec.subject}の{spec.project_type}で、過去に参考になる社内事例はありますか？"
                "実施条件や注意点も教えてください。"
            )
            gold_ids = matching_ids
        gold_references = [
            {
                "report_id": rid,
                "dept": spec_by_report_id[rid].dept,
                "evidence": spec_by_report_id[rid].lesson,
            }
            for rid in sorted(gold_ids)
        ]
        qa_pairs.append(
            {
                "qa_id": f"QA-{i + 1:03d}",
                "question": question,
                "asking_dept": asking_dept,
                "cross_dept": cross_dept,
                "gold_references": gold_references,
            }
        )
    return qa_pairs


def _write_eval_qa(qa_pairs: list[dict], out_dir: Path) -> None:
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "qa_pairs.json"
    path.write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated: {path}")


# --- CLI --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="部署横断プロジェクト知見・教訓のダミーレポート合成データ生成"
    )
    parser.add_argument("--count", type=int, default=60, help="生成するレポート件数")
    parser.add_argument("--eval-count", type=int, default=15, help="生成する評価QAペア件数")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（再現性のため固定）")
    parser.add_argument("--out-dir", type=Path, default=SYNTH_REPORTS_DIR)
    parser.add_argument("--eval-out-dir", type=Path, default=EVAL_DIR)
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count は1以上を指定してください。")
    if args.eval_count < 1:
        raise SystemExit("--eval-count は1以上を指定してください。")

    specs = generate_report_specs(args.count, seed=args.seed)

    config = load_config()
    with LLMClient(config.ai) as client:
        if not client.ping():
            raise SystemExit(
                f"LM Studio ({config.ai.base_url}) に接続できません。起動してモデルをロードしてください。"
            )
        last_error: LLMConnectionError | None = None
        for attempt in range(1, _MAX_BATCH_ATTEMPTS + 1):
            try:
                generate_reports(client, specs, args.out_dir)
                break
            except LLMConnectionError as exc:
                last_error = exc
                print(
                    f"バッチ全体の生成に失敗しました（{attempt}/{_MAX_BATCH_ATTEMPTS}回目）: {exc}\n"
                    "ローカルLLMのサンプリングのブレによる一時的な失敗のことが多いため、"
                    "自動的に最初からやり直します。"
                )
        else:
            assert last_error is not None
            raise SystemExit(
                f"{_MAX_BATCH_ATTEMPTS}回試しましたが生成に失敗しました: {last_error}\n"
                "ロードしているモデルを変えるか、しばらく時間を置いて再実行してください。"
            )

    qa_pairs = generate_eval_qa(specs, args.eval_count, seed=args.seed)
    _write_eval_qa(qa_pairs, args.eval_out_dir)


if __name__ == "__main__":
    main()
