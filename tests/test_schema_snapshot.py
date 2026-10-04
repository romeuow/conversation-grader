import json
from pathlib import Path

from grader.models import CriterionGrade

SNAPSHOT = Path(__file__).parent / "snapshots" / "criterion_grade.schema.json"


def test_criterion_grade_schema_matches_snapshot():
    """The structured-output contract sent to the LLM must not change silently.

    Regenerate with: uv run python -c "import json; from grader.models import CriterionGrade; \
print(json.dumps(CriterionGrade.model_json_schema(), indent=2, ensure_ascii=False))" > \
tests/snapshots/criterion_grade.schema.json
    """
    current = CriterionGrade.model_json_schema()
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert current == expected


def test_schema_shape():
    schema = CriterionGrade.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"score", "rationale"}
    assert schema["properties"]["score"]["minimum"] == 0
    assert schema["properties"]["score"]["maximum"] == 5
    assert schema["properties"]["evidence"]["type"] == "array"
