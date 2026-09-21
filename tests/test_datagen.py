"""datagen（DAGノードA）の純粋ロジックのテスト。LLM呼び出しは行わない。"""

from __future__ import annotations

import pytest

from cae_rag.datagen import (
    ANALYSIS_TYPES,
    DEPARTMENTS,
    DEPT_TERMINOLOGY,
    FILE_FORMATS,
    METADATA_FIELDS,
    RESULT_IMAGE_SECTION,
    ReportSpec,
    _generate_one_report,
    _metadata_dict,
    _pdf_lines,
    _split_sections,
    _validate_report_body,
    build_prompt,
    generate_eval_qa,
    generate_report_specs,
)
from cae_rag.llm_client import LLMConnectionError


def test_generate_report_specs_deterministic_with_seed():
    a = generate_report_specs(30, seed=1)
    b = generate_report_specs(30, seed=1)
    assert [s.report_id for s in a] == [s.report_id for s in b]
    assert [(s.dept, s.analysis_type, s.part) for s in a] == [(s.dept, s.analysis_type, s.part) for s in b]


def test_generate_report_specs_covers_all_dept_analysis_combos():
    n_combos = len(DEPARTMENTS) * len(ANALYSIS_TYPES)
    specs = generate_report_specs(n_combos, seed=7)
    seen = {(s.dept, s.analysis_type) for s in specs}
    assert len(seen) == n_combos


def test_generate_report_specs_parts_belong_to_their_department():
    specs = generate_report_specs(50, seed=3)
    for s in specs:
        assert s.part in DEPARTMENTS[s.dept]


def _make_spec(**overrides) -> ReportSpec:
    from datetime import date

    base = dict(
        report_id="RPT-001",
        dept="ボディ設計部",
        analysis_type="静解析（線形）",
        part="フロントドアパネル",
        solver="Abaqus",
        material="高張力鋼板(980MPa級)",
        failure_mode="メッシュが粗く、応力集中部を捉えられていなかった",
        author="担当者A",
        report_date=date(2023, 1, 1),
        file_format="md",
    )
    base.update(overrides)
    return ReportSpec(**base)


def _required_headings(spec: ReportSpec) -> list[str]:
    term = DEPT_TERMINOLOGY[spec.dept]
    return [
        "解析目的",
        "対象部品・製品カテゴリ",
        term["conditions"],
        term["mesh"],
        "材料物性",
        "使用ソルバー",
        "結果サマリー",
        "トラブルシューティング・教訓",
    ]


def _valid_body(spec: ReportSpec) -> str:
    return "\n\n".join(f"## {h}\n本文がここに入ります。" for h in _required_headings(spec))


def test_build_prompt_lists_required_headings_in_user_prompt():
    spec = _make_spec()
    _, user = build_prompt(spec)
    for heading in _required_headings(spec):
        assert f"## {heading}" in user


def test_validate_report_body_accepts_valid_body():
    spec = _make_spec()
    _validate_report_body(spec, _valid_body(spec))  # 例外が出なければOK


def test_validate_report_body_rejects_missing_section():
    spec = _make_spec()
    headings = _required_headings(spec)
    body = "\n\n".join(f"## {h}\n本文" for h in headings[:-1])  # 最後の見出しを欠落させる
    with pytest.raises(LLMConnectionError):
        _validate_report_body(spec, body)


def test_validate_report_body_rejects_duplicate_heading():
    spec = _make_spec()
    headings = _required_headings(spec)
    body = "\n\n".join(f"## {h}\n本文" for h in [*headings, headings[-1]])  # 最後を重複させる
    with pytest.raises(LLMConnectionError):
        _validate_report_body(spec, body)


def test_validate_report_body_rejects_empty_section():
    spec = _make_spec()
    headings = _required_headings(spec)
    parts = [f"## {h}\n本文" for h in headings[:-1]]
    parts.append(f"## {headings[-1]}\n")  # 最後のセクションを空にする
    body = "\n\n".join(parts)
    with pytest.raises(LLMConnectionError):
        _validate_report_body(spec, body)


def test_generate_eval_qa_gold_references_always_include_the_source_spec():
    specs = generate_report_specs(40, seed=11)
    qa_pairs = generate_eval_qa(specs, 10, seed=11)
    assert len(qa_pairs) == 10
    for qa in qa_pairs:
        assert len(qa["gold_references"]) >= 1
        assert "question" in qa and qa["question"]


def test_generate_eval_qa_cross_dept_excludes_asking_department():
    """退行テスト: 以前、部署をまたいだ設問(cross_dept)で、質問者自身の部署のレポートが
    誤って正解に含まれるバグがあった（codexレビューで発見・修正済み）。"""
    specs = generate_report_specs(60, seed=2)  # このseedで実際に問題が再現していた
    qa_pairs = generate_eval_qa(specs, 15, seed=2)
    dept_by_id = {s.report_id: s.dept for s in specs}
    for qa in qa_pairs:
        if not qa["cross_dept"]:
            continue
        for ref in qa["gold_references"]:
            assert ref["dept"] != qa["asking_dept"], (
                f"{qa['qa_id']}: 他部署の事例を求める設問なのに、質問者自身の部署"
                f"（{qa['asking_dept']}）のレポート {ref['report_id']} が正解に含まれている"
            )


def test_generate_eval_qa_same_dept_question_gold_dept_matches_asking_dept():
    specs = generate_report_specs(30, seed=5)
    qa_pairs = generate_eval_qa(specs, 8, seed=5)
    for qa in qa_pairs:
        if qa["cross_dept"]:
            continue
        # 自部署内の設問では、asking_deptは出典レポートの部署のいずれかと一致するはず
        # （同一(部品,解析種別)の中には他部署のレポートも混ざりうるため、asking_dept自身の
        # レポートが正解集合に含まれることだけを確認する）。
        assert any(ref["dept"] == qa["asking_dept"] for ref in qa["gold_references"])


class _ScriptedClient:
    """.chatが呼ばれるたびに、あらかじめ用意した本文を順番に返すフェイク。
    実際にLM Studioに繋がっていないと再現しづらい「検証失敗→リトライで成功」
    パターンをテストするために使う。"""

    def __init__(self, bodies: list[str]):
        self._bodies = list(bodies)
        self.call_count = 0

    def chat(self, *args, **kwargs) -> str:
        self.call_count += 1
        return self._bodies.pop(0)


def test_generate_one_report_retries_after_validation_failure():
    # _generate_result_image()はmatplotlib依存で、このテストの関心（リトライ挙動）とは
    # 無関係なので、ここだけ差し替えてmatplotlibなしでも検証できるようにする。
    import cae_rag.datagen as datagen_module

    spec = _make_spec()
    headings = _required_headings(spec)
    invalid_body = "\n\n".join(f"## {h}\n本文" for h in headings[:-1])  # 1見出し欠落
    valid_body = _valid_body(spec)

    client = _ScriptedClient([invalid_body, valid_body])
    original = datagen_module._generate_result_image
    datagen_module._generate_result_image = lambda spec: b"fake-image-bytes"
    try:
        metadata, sections, image = _generate_one_report(client, spec)
    finally:
        datagen_module._generate_result_image = original

    assert client.call_count == 2  # 1回目失敗、2回目で成功
    assert metadata["report_id"] == spec.report_id
    assert sections[-1][1] == "本文がここに入ります。"
    assert image == b"fake-image-bytes"


def test_generate_one_report_raises_after_exhausting_retries():
    spec = _make_spec()
    headings = _required_headings(spec)
    invalid_body = "\n\n".join(f"## {h}\n本文" for h in headings[:-1])

    client = _ScriptedClient([invalid_body, invalid_body, invalid_body, "この4回目は呼ばれないはず"])
    with pytest.raises(LLMConnectionError):
        _generate_one_report(client, spec)

    assert client.call_count == 3  # _MAX_GENERATION_ATTEMPTS=3で打ち切られる


def test_generate_report_specs_assigns_file_format_from_pool():
    specs = generate_report_specs(40, seed=13)
    assert all(s.file_format in FILE_FORMATS for s in specs)
    # 部署に固定しない設計なので、十分な件数があれば複数の形式が混在するはず。
    assert len({s.file_format for s in specs}) > 1


def test_metadata_dict_has_all_expected_fields_in_order():
    spec = _make_spec()
    meta = _metadata_dict(spec)
    assert list(meta.keys()) == METADATA_FIELDS
    assert meta["report_id"] == spec.report_id
    assert meta["date"] == spec.report_date.isoformat()


def test_split_sections_matches_body_structure():
    spec = _make_spec()
    body = _valid_body(spec)
    sections = _split_sections(body)
    assert [h for h, _ in sections] == _required_headings(spec)
    assert all(text == "本文がここに入ります。" for _, text in sections)


def test_result_image_section_matches_ingest_constant():
    """退行テスト: datagenとingestは互いに依存させない設計上、画像を添付する
    セクション名を別々の定数として持っている。ズレるとVLMキャプションが
    正しいセクションに合流しなくなるため、一致していることを保証する。"""
    from cae_rag.ingest import RESULT_IMAGE_SECTION as INGEST_RESULT_IMAGE_SECTION

    assert RESULT_IMAGE_SECTION == INGEST_RESULT_IMAGE_SECTION
    assert RESULT_IMAGE_SECTION in _required_headings(_make_spec())


def test_pdf_lines_round_trips_through_ingest_parser():
    """datagen._pdf_lines() の出力を、ingest._parse_pdf_text() でそのまま
    パースし直せることを確認する（pypdf/reportlabなしで往復ロジックだけ検証する）。"""
    from cae_rag.ingest import _parse_pdf_text

    spec = _make_spec()
    metadata = _metadata_dict(spec)
    sections = [("解析目的", "1行目の本文です。"), ("結果サマリー", "こちらも短い本文。")]

    lines = _pdf_lines(metadata, sections)
    full_text = "\n".join(lines)
    parsed_meta, parsed_sections = _parse_pdf_text(full_text)

    assert parsed_meta == metadata
    assert [h for h, _ in parsed_sections] == ["解析目的", "結果サマリー"]
    assert parsed_sections[0][1] == "1行目の本文です。"
    assert parsed_sections[1][1] == "こちらも短い本文。"
