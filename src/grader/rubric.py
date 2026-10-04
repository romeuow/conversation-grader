"""Rubric loading and validation."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RubricError(ValueError):
    """Raised when a rubric file is missing or invalid."""


class Criterion(BaseModel):
    """A single grading criterion with a 0-5 anchored scale."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,40}$")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    weight: float = Field(gt=0, le=10)
    anchors: dict[int, str]

    @field_validator("anchors")
    @classmethod
    def _validate_anchors(cls, anchors: dict[int, str]) -> dict[int, str]:
        if not anchors:
            raise ValueError("anchors must not be empty")
        for level in anchors:
            if not 0 <= level <= 5:
                raise ValueError(f"anchor level {level} outside 0-5")
        if 0 not in anchors or 5 not in anchors:
            raise ValueError("anchors must define at least levels 0 and 5")
        return dict(sorted(anchors.items()))


class Rubric(BaseModel):
    """A versioned set of criteria."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,40}$")
    version: str = Field(min_length=1)
    name: str
    description: str = ""
    criteria: list[Criterion] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Rubric:
        ids = [c.id for c in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion ids must be unique")
        return self

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria)

    def get(self, criterion_id: str) -> Criterion:
        for criterion in self.criteria:
            if criterion.id == criterion_id:
                return criterion
        raise KeyError(criterion_id)


def load_rubric(path: Path) -> Rubric:
    """Load and validate a rubric YAML file."""
    if not path.exists():
        raise RubricError(f"rubric file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RubricError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RubricError(f"rubric root must be a mapping: {path}")
    try:
        return Rubric.model_validate(raw)
    except ValueError as exc:
        raise RubricError(f"invalid rubric {path}: {exc}") from exc


class RubricRegistry:
    """Loads every `*.yaml` under a directory and indexes rubrics by id."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._rubrics: dict[str, Rubric] = {}
        for file in sorted(directory.glob("*.yaml")):
            rubric = load_rubric(file)
            if rubric.id in self._rubrics:
                raise RubricError(f"duplicate rubric id {rubric.id!r} in {file}")
            self._rubrics[rubric.id] = rubric

    def get(self, rubric_id: str) -> Rubric:
        try:
            return self._rubrics[rubric_id]
        except KeyError as exc:
            raise RubricError(f"unknown rubric {rubric_id!r}") from exc

    def list(self) -> list[Rubric]:
        return list(self._rubrics.values())

    def __contains__(self, rubric_id: str) -> bool:
        return rubric_id in self._rubrics
