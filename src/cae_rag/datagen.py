"""合成データ生成（DAGノードA）。

自動車部品の構造解析（FEM）に関する、社内ナレッジ検索RAGのデモ用ダミーレポートを生成する。

前提（プラン参照）:
    - 実在・架空を問わず企業名は一切出さない。「ある1社内の複数部署」という匿名設定。
    - 部署間で情報共有が完全には統一されていない状況を再現するため、部署ごとに
      見出し語彙（ハウススタイル）を微妙に変える。
    - ファイル形式（Markdown/Word/Excel/PowerPoint/PDF）はレポートごとにランダムに
      割り当てる（部署には固定しない。現場でファイル形式が混在している状況の再現）。
    - 対象は構造解析（FEM）のみ。熱解析・CFD等は対象外。

生成はローカルLLM（LM Studio等、OpenAI互換API）経由。`python -m cae_rag.datagen` で実行する。
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

# --- ドメイン語彙 -----------------------------------------------------------

ANALYSIS_TYPES = [
    "静解析（線形）",
    "固有値・モーダル解析",
    "非線形静解析（大変形・接触）",
    "疲労解析",
    "振動解析（強制応答）",
]

SOLVERS = ["Abaqus", "Ansys", "Nastran"]

DEPARTMENTS: dict[str, list[str]] = {
    "ボディ設計部": ["フロントドアパネル", "リアフェンダー", "ルーフパネル", "フロントバンパービーム", "ボンネット(フード)"],
    "シャシー設計部": ["フロントサブフレーム", "リアクロスメンバー", "トーコントロールアーム", "スタビライザーリンク", "エンジンマウントブラケット"],
    "パワートレイン設計部": ["トランスミッションマウント", "排気系サポートブラケット", "オイルパン", "インタークーラーステー", "プロペラシャフトサポート"],
    "ブレーキ・サスペンション設計部": ["ブレーキキャリパーブラケット", "サスペンションロアアーム", "コイルスプリングシート", "ダンパーマウント", "ナックル"],
    "品質保証部": ["フロントドアパネル", "リアクロスメンバー", "ブレーキキャリパーブラケット", "トランスミッションマウント", "コイルスプリングシート"],
}

# 部署ごとのハウススタイル（見出し語彙の揺れ）。部署間で用語が統一されていない状況を再現する。
DEPT_TERMINOLOGY: dict[str, dict[str, str]] = {
    "ボディ設計部": {"conditions": "解析条件", "mesh": "メッシュ設定"},
    "シャシー設計部": {"conditions": "荷重・拘束条件", "mesh": "メッシュ諸元"},
    "パワートレイン設計部": {"conditions": "境界条件", "mesh": "要素分割"},
    "ブレーキ・サスペンション設計部": {"conditions": "拘束・荷重条件", "mesh": "メッシュ条件"},
    "品質保証部": {"conditions": "入力条件", "mesh": "メッシュ仕様"},
}

# ファイル形式は部署に固定せず、レポートごとにランダムに割り当てる（どの部署でも
# 複数の形式が混在しうる）。部署間の非統一性は用語（DEPT_TERMINOLOGY）側で表現する。
FILE_FORMATS = ["md", "docx", "xlsx", "pdf", "pptx"]

# メタデータの固定フィールド順。Word/Excel/PowerPoint/PDFの書き出しで共通して使う。
METADATA_FIELDS = ["report_id", "dept", "analysis_type", "part", "solver", "material", "author", "date"]

FAILURE_MODES = [
    "メッシュが粗く、応力集中部を捉えられていなかった",
    "収束計算が発散し、時間刻みを細分化して収束させた",
    "材料の弾塑性物性値の入力単位を誤り、結果が過大評価された",
    "拘束条件が実機と乖離しており、剛体モードが残ってしまった",
    "接触条件の摩擦係数設定が不適切で、応力分布が不自然になった",
    "要素タイプ（1次/2次要素）の選定が不適切で、曲げ剛性を過小評価した",
    "荷重ケースの組み合わせに漏れがあり、再解析が必要になった",
    "疲労評価のS-N線図の適用範囲を誤り、寿命予測が過大だった",
    "共振点が想定荷重周波数と近接しており、追加補強が必要だった",
    "解析結果と実機評価（試作品での振動試験）に乖離があり、境界条件を見直した",
]

MATERIALS = ["高張力鋼板(980MPa級)", "アルミニウム合金(A6061)", "冷間圧延鋼板(SPCC)", "アルミダイカスト(ADC12)", "炭素繊維強化樹脂(CFRP)"]

AUTHOR_NAMES = [f"担当者{c}" for c in "ABCDEFGHIJ"]


@dataclass
class ReportSpec:
    report_id: str
    dept: str
    analysis_type: str
    part: str
    solver: str
    material: str
    failure_mode: str
    author: str
    report_date: date
    file_format: str


def generate_report_specs(count: int, *, seed: int | None = None) -> list[ReportSpec]:
    """部署・解析種別を横断的にカバーするようレポート仕様を生成する（層化サンプリング）。"""
    rng = random.Random(seed)
    depts = list(DEPARTMENTS.keys())
    combos = [(d, a) for d in depts for a in ANALYSIS_TYPES]
    rng.shuffle(combos)

    specs: list[ReportSpec] = []
    base_date = date(2023, 1, 1)
    for i in range(count):
        dept, analysis_type = combos[i % len(combos)]
        part = rng.choice(DEPARTMENTS[dept])
        specs.append(
            ReportSpec(
                report_id=f"RPT-{i + 1:03d}",
                dept=dept,
                analysis_type=analysis_type,
                part=part,
                solver=rng.choice(SOLVERS),
                material=rng.choice(MATERIALS),
                failure_mode=rng.choice(FAILURE_MODES),
                author=rng.choice(AUTHOR_NAMES),
                report_date=base_date + timedelta(days=rng.randint(0, 900)),
                file_format=rng.choice(FILE_FORMATS),
            )
        )
    return specs


def build_prompt(spec: ReportSpec) -> tuple[str, str]:
    term = DEPT_TERMINOLOGY[spec.dept]
    system = (
        "あなたは自動車部品メーカーの構造解析(FEM)エンジニアです。"
        "実在・架空を問わず企業名や具体的なブランド名は一切書かないでください。"
        "社内向けの解析レポートをMarkdown形式で日本語で書いてください。"
        "数値は具体的な架空の値を使ってよいですが、実在製品の公表値を模倣しないでください。"
    )
    user = f"""以下の条件で、社内の構造解析レポート本文をMarkdown形式で書いてください。

- 作成部署: {spec.dept}
- 対象部品: {spec.part}
- 解析種別: {spec.analysis_type}
- 使用ソルバー: {spec.solver}
- 主要材料: {spec.material}
- 担当者: {spec.author}
- 日付: {spec.report_date.isoformat()}
- 今回の解析で実際に起きたトラブル・教訓: {spec.failure_mode}

以下の見出し構成に厳密に従ってください（各見出しは `## ` で始める）:

## 解析目的
## 対象部品・製品カテゴリ
## {term["conditions"]}
## {term["mesh"]}
## 材料物性
## 使用ソルバー
## 結果サマリー
## トラブルシューティング・教訓

「トラブルシューティング・教訓」セクションには、上記の「実際に起きたトラブル・教訓」を具体的な数値・状況付きで詳しく書いてください。
各セクションは3〜6行程度の具体的な記述にしてください。タイトル行（# で始まる見出し）は書かず、上記の `## ` 見出しから書き始めてください。"""
    return system, user


def _metadata_dict(spec: ReportSpec) -> dict[str, str]:
    return {
        "report_id": spec.report_id,
        "dept": spec.dept,
        "analysis_type": spec.analysis_type,
        "part": spec.part,
        "solver": spec.solver,
        "material": spec.material,
        "author": spec.author,
        "date": spec.report_date.isoformat(),
    }


_SECTION_HEADING_RE = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)


def _split_sections(body: str) -> list[tuple[str, str]]:
    """本文を `## 見出し` 単位で (見出し, 本文) のリストに分割する。

    ingest.split_into_sections() と同じ考え方だが、datagenがingestに依存する
    （DAGの向きと逆の結合が生じる）のを避けるため、ここで独自に持つ。
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
    """生成された本文が、ingest側が期待する `## 見出し` 構成を満たしているか検証する。

    見出しの level（`##`）や語彙がずれていたり、本文が空のセクションがあると、
    ingest.split_into_sections() が対象セクションを拾えず検索対象から漏れてしまう。
    そのまま前回の正常なデータセットを上書きしないよう、ここで必ず弾く。
    """
    term = DEPT_TERMINOLOGY[spec.dept]
    required = [
        "解析目的",
        "対象部品・製品カテゴリ",
        term["conditions"],
        term["mesh"],
        "材料物性",
        "使用ソルバー",
        "結果サマリー",
        "トラブルシューティング・教訓",
    ]
    matches = list(_SECTION_HEADING_RE.finditer(body))
    headings = [m.group(1).strip() for m in matches]

    missing = [h for h in required if h not in headings]
    if missing:
        raise LLMConnectionError(f"{spec.report_id}: 生成レポートに必須セクションが欠けています: {missing}")

    # 同じ見出しが重複すると、ingest側でチャンクIDが衝突してChromaへの格納に失敗する。
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


def _generate_one_report(client: LLMClient, spec: ReportSpec) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """1件のレポートを生成し、(メタデータ, [(見出し, 本文), ...]) を返す。

    小型のローカルLLMは、指定した見出し構成を毎回厳密には守れないことがある
    （実際に60件中1件、見出し欠落で失敗する事例が起きた）。温度付き(0.7)サンプリング
    なので同じ入力でも生成のたびに結果が変わることを利用し、生成→検証に失敗したら
    数回リトライしてから諦める。

    ここではまだファイルには書き出さない（書式はspec.file_formatによって異なり、
    実際の書き出しはgenerate_reports()が全件成功を確認してから行う）。
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
        return _metadata_dict(spec), _split_sections(body)

    assert last_error is not None
    raise last_error


# --- フォーマット別の書き出し ------------------------------------------------


def _report_title(metadata: dict[str, str]) -> str:
    return f"{metadata['part']} {metadata['analysis_type']} 解析レポート（{metadata['report_id']}）"


def _write_markdown(metadata: dict[str, str], sections: list[tuple[str, str]], path: Path) -> None:
    lines = ["---", *(f"{key}: {metadata[key]}" for key in METADATA_FIELDS), "---", ""]
    header = "\n".join(lines)
    title = f"# {_report_title(metadata)}\n\n"
    body = "\n\n".join(f"## {heading}\n{text}" for heading, text in sections) + "\n"
    path.write_text(header + title + body, encoding="utf-8")


def _write_docx(metadata: dict[str, str], sections: list[tuple[str, str]], path: Path) -> None:
    from docx import Document

    doc = Document()
    doc.add_heading(_report_title(metadata), level=1)

    table = doc.add_table(rows=len(METADATA_FIELDS), cols=2)
    for row, key in zip(table.rows, METADATA_FIELDS, strict=True):
        row.cells[0].text = key
        row.cells[1].text = metadata[key]

    for heading, text in sections:
        doc.add_heading(heading, level=2)
        doc.add_paragraph(text)

    doc.save(str(path))


def _write_xlsx(metadata: dict[str, str], sections: list[tuple[str, str]], path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "report"
    for i, key in enumerate(METADATA_FIELDS, start=1):
        ws.cell(row=i, column=1, value=key)
        ws.cell(row=i, column=2, value=metadata[key])

    start_row = len(METADATA_FIELDS) + 2  # メタデータの後に1行空ける
    for offset, (heading, text) in enumerate(sections):
        ws.cell(row=start_row + offset, column=1, value=heading)
        ws.cell(row=start_row + offset, column=2, value=text)

    wb.save(str(path))


def _write_pptx(metadata: dict[str, str], sections: list[tuple[str, str]], path: Path) -> None:
    from pptx import Presentation

    prs = Presentation()
    layout = prs.slide_layouts[1]  # タイトル + コンテンツ

    meta_slide = prs.slides.add_slide(layout)
    meta_slide.shapes.title.text = "レポートメタデータ"
    meta_slide.placeholders[1].text_frame.text = "\n".join(f"{key}: {metadata[key]}" for key in METADATA_FIELDS)

    for heading, text in sections:
        slide = prs.slides.add_slide(layout)
        slide.shapes.title.text = heading
        slide.placeholders[1].text_frame.text = text

    prs.save(str(path))


def _pdf_lines(metadata: dict[str, str], sections: list[tuple[str, str]]) -> list[str]:
    """PDFページに描画するテキスト行を組み立てる（reportlabに依存しない純粋関数）。

    ingest._parse_pdf_text() はこの行の並びを前提にパースする。reportlab/pypdf
    どちらもインストールされていない環境でも、この関数とingest側のパーサーだけで
    往復（書く→読む）ロジックの整合性をテストできるように、描画処理と分離してある。
    """
    import textwrap

    # 日本語はreportlabのdrawStringが自動折り返ししないため、あらかじめ
    # 全角換算で1行38文字程度に折り返してから描画する。
    lines: list[str] = [f"{key}: {metadata[key]}" for key in METADATA_FIELDS]
    lines.append("")
    for heading, text in sections:
        lines.append(f"## {heading}")
        for raw_line in text.splitlines() or [""]:
            lines.extend(textwrap.wrap(raw_line, width=38) or [""])
        lines.append("")
    return lines


def _write_pdf(metadata: dict[str, str], sections: list[tuple[str, str]], path: Path) -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    font_name = "HeiseiKakuGo-W5"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))

    lines = _pdf_lines(metadata, sections)

    page_width, page_height = 595, 842  # A4 (pt)
    left_margin, top_margin, bottom_margin = 40, 800, 40
    font_size, line_height = 11, 16

    c = canvas.Canvas(str(path), pagesize=(page_width, page_height))
    y = top_margin
    c.setFont(font_name, font_size)
    for line in lines:
        if y < bottom_margin:
            c.showPage()
            c.setFont(font_name, font_size)
            y = top_margin
        c.drawString(left_margin, y, line)
        y -= line_height
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

    # 全件のLLM生成が完了してからディスクに書き出す。途中のリクエストが失敗しても
    # 前回のデータセット（およびそれと対応するqa_pairs.json）を壊さないため。
    generated: dict[str, tuple[dict[str, str], list[tuple[str, str]]]] = {}
    for spec in specs:
        generated[spec.report_id] = _generate_one_report(client, spec)
        print(f"generated (in-memory): {spec.report_id}")

    # ここまで来て初めて、前回の生成物を一掃して書き出す。--count を減らして
    # 再実行したときに古いレポートが残り、今回のQAペアと矛盾したデータセットに
    # なるのを防ぐ（対象は今回使いうる全フォーマットの拡張子）。
    for fmt in FILE_FORMATS:
        for stale in out_dir.glob(f"RPT-*.{fmt}"):
            stale.unlink()
    for spec in specs:
        metadata, sections = generated[spec.report_id]
        path = out_dir / f"{spec.report_id}.{spec.file_format}"
        _WRITERS[spec.file_format](metadata, sections, path)
        print(f"wrote: {path}")


# --- 評価用 gold-standard QAペア -------------------------------------------


def generate_eval_qa(specs: list[ReportSpec], count: int, *, seed: int | None = None) -> list[dict]:
    """評価用QAペアを作る。一部は「別部署の過去事例を知らずに質問するケース」を含める。

    実際のLLM呼び出しはせず、レポート仕様から機械的に問いと正解根拠を組み立てる
    （gold-standardは人手検証可能な単純な形にしておく）。
    """
    rng = random.Random(seed)
    chosen = rng.sample(specs, k=min(count, len(specs)))

    # 設問文には部品名と解析種別しか出てこないため、同じ(部品, 解析種別)の組み合わせを
    # 持つレポートが複数あると、正解が1件だけだと決め打ちできない（どれも妥当な参照先）。
    # そのため正解は「同じ組み合わせを持つ全レポートの一覧（各々のdept/evidence付き）」
    # として持たせる（1件のdept/evidenceで代表させると、他の正解と矛盾する）。
    reports_by_key: dict[tuple[str, str], list[str]] = {}
    spec_by_report_id: dict[str, ReportSpec] = {}
    for s in specs:
        reports_by_key.setdefault((s.part, s.analysis_type), []).append(s.report_id)
        spec_by_report_id[s.report_id] = s

    qa_pairs: list[dict] = []
    for i, spec in enumerate(chosen):
        cross_dept = i % 3 == 0  # 3件に1件は部署をまたいだ想定の設問にする
        matching_ids = reports_by_key[(spec.part, spec.analysis_type)]
        if cross_dept:
            other_depts = [d for d in DEPARTMENTS if d != spec.dept]
            asking_dept = rng.choice(other_depts)
            question = (
                f"{asking_dept}です。{spec.part}に似た部品で{spec.analysis_type}を検討しています。"
                f"他部署で参考になりそうな過去の{spec.analysis_type}の事例はありますか？"
                "特に気をつけるべき落とし穴があれば教えてください。"
            )
            # 「他部署の事例」を明示的に求めている設問なので、質問者自身の部署の
            # レポートは正解から除く（同一(部品,解析種別)でも自部署のものは対象外）。
            gold_ids = [r for r in matching_ids if spec_by_report_id[r].dept != asking_dept]
        else:
            asking_dept = spec.dept
            question = (
                f"{spec.part}の{spec.analysis_type}で、過去に参考になる社内事例はありますか？"
                "解析条件や注意点も教えてください。"
            )
            gold_ids = matching_ids
        gold_references = [
            {
                "report_id": rid,
                "dept": spec_by_report_id[rid].dept,
                "evidence": spec_by_report_id[rid].failure_mode,
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
    parser = argparse.ArgumentParser(description="CAE構造解析ダミーレポートの合成データ生成")
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
    with LLMClient(config.llm) as client:
        if not client.ping():
            raise SystemExit(
                f"LM Studio ({config.llm.base_url}) に接続できません。起動してモデルをロードしてください。"
            )
        generate_reports(client, specs, args.out_dir)

    qa_pairs = generate_eval_qa(specs, args.eval_count, seed=args.seed)
    _write_eval_qa(qa_pairs, args.eval_out_dir)


if __name__ == "__main__":
    main()
