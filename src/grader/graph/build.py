"""Graph assembly and a thin facade used by the API, the CLI and the evals."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from grader.config import Settings, get_settings
from grader.graph.nodes import GraphNodes, Pricing
from grader.graph.state import GraderState
from grader.llm.base import LLMClient
from grader.models import Conversation, ConversationResult
from grader.rubric import Rubric


def build_graph(
    llm: LLMClient,
    rubric: Rubric,
    *,
    pricing: Pricing | None = None,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Compile the grading graph.

    normalize -> redact_pii -> [grade_criterion x N] -> detect_risks -> aggregate -> finalize
    """
    if pricing is None:
        settings = get_settings()
        pricing = Pricing(settings.llm_input_price_per_mtok, settings.llm_output_price_per_mtok)
    nodes = GraphNodes(llm, rubric, pricing)

    graph = StateGraph(GraderState)
    graph.add_node("normalize", nodes.normalize)
    graph.add_node("redact_pii", nodes.redact_pii)
    graph.add_node("grade_criterion", nodes.grade_criterion)
    graph.add_node("detect_risks", nodes.detect_risks)
    graph.add_node("aggregate", nodes.aggregate)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "redact_pii")
    graph.add_conditional_edges("redact_pii", nodes.fan_out_criteria, ["grade_criterion"])
    graph.add_edge("grade_criterion", "detect_risks")
    graph.add_edge("detect_risks", "aggregate")
    graph.add_edge("aggregate", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer, name=f"grader-{rubric.id}")


class ConversationGrader:
    """Facade: grade a `Conversation` and get a `ConversationResult` back."""

    def __init__(
        self,
        llm: LLMClient,
        rubric: Rubric,
        *,
        settings: Settings | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        settings = settings or get_settings()
        self.rubric = rubric
        self.llm = llm
        self._checkpointer = checkpointer
        self.graph = build_graph(
            llm,
            rubric,
            pricing=Pricing(settings.llm_input_price_per_mtok, settings.llm_output_price_per_mtok),
            checkpointer=checkpointer,
        )

    def _config(self, conversation: Conversation) -> dict[str, Any] | None:
        if self._checkpointer is None:
            return None
        return {"configurable": {"thread_id": conversation.id}}

    def grade(self, conversation: Conversation) -> ConversationResult:
        state = self.graph.invoke({"conversation": conversation}, config=self._config(conversation))
        return state["result"]

    async def agrade(self, conversation: Conversation) -> ConversationResult:
        state = await self.graph.ainvoke(
            {"conversation": conversation}, config=self._config(conversation)
        )
        return state["result"]
