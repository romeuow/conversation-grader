from pathlib import Path

import pytest

from grader.rubric import RubricError, RubricRegistry, load_rubric


def test_load_sales_rubric(rubric):
    assert rubric.id == "sales_v1"
    assert [c.id for c in rubric.criteria] == [
        "opening",
        "discovery",
        "objection_handling",
        "next_steps",
        "compliance",
    ]
    assert rubric.total_weight == pytest.approx(7.0)
    assert rubric.get("compliance").weight == 2.0
    assert list(rubric.get("opening").anchors) == [0, 1, 3, 5]


def test_get_unknown_criterion(rubric):
    with pytest.raises(KeyError):
        rubric.get("nope")


def test_missing_file(tmp_path: Path):
    with pytest.raises(RubricError, match="not found"):
        load_rubric(tmp_path / "missing.yaml")


def test_invalid_yaml(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("id: [unclosed", encoding="utf-8")
    with pytest.raises(RubricError, match="invalid YAML"):
        load_rubric(path)


def test_root_must_be_mapping(tmp_path: Path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(RubricError, match="mapping"):
        load_rubric(path)


@pytest.mark.parametrize(
    "anchors, message",
    [
        ("{}", "anchors must not be empty"),
        ("{0: a, 7: b}", "outside 0-5"),
        ("{1: a, 5: b}", "levels 0 and 5"),
    ],
)
def test_anchor_validation(tmp_path: Path, anchors: str, message: str):
    path = tmp_path / "r.yaml"
    path.write_text(
        f"""
id: r_v1
version: "1"
name: R
criteria:
  - {{id: c1, name: C, description: D, weight: 1, anchors: {anchors}}}
""",
        encoding="utf-8",
    )
    with pytest.raises(RubricError, match=message):
        load_rubric(path)


def test_duplicate_criterion_ids(tmp_path: Path):
    path = tmp_path / "dup.yaml"
    path.write_text(
        """
id: dup_v1
version: "1"
name: Dup
criteria:
  - {id: c1, name: C, description: D, weight: 1, anchors: {0: a, 5: b}}
  - {id: c1, name: C, description: D, weight: 1, anchors: {0: a, 5: b}}
""",
        encoding="utf-8",
    )
    with pytest.raises(RubricError, match="unique"):
        load_rubric(path)


def test_registry(registry):
    assert "sales_v1" in registry
    assert registry.get("sales_v1").version == "1.0.0"
    assert [r.id for r in registry.list()] == ["sales_v1"]
    with pytest.raises(RubricError, match="unknown rubric"):
        registry.get("other")


def test_registry_rejects_duplicate_ids(tmp_path: Path, tmp_rubric_file: Path):
    (tmp_path / "copy.yaml").write_text(tmp_rubric_file.read_text(), encoding="utf-8")
    with pytest.raises(RubricError, match="duplicate rubric id"):
        RubricRegistry(tmp_path)
