from __future__ import annotations

from llm.client import LLMClient, Message
from models import QuestionPair, ScaffoldResult, TraceStep
from scaffolds.base import extract_final_answer


def run_control(
    pair: QuestionPair, client: LLMClient, trial: int, temperature: float = 1.0
) -> ScaffoldResult:
    prompt = (
        f"Answer the following question. End with a line starting with 'Final answer:'.\n\n"
        f"Question: {pair.x}"
    )
    completion = client.complete(
        messages=[Message(role="user", content=prompt)],
        max_tokens=1024,
        temperature=temperature,
    )
    trace = [
        TraceStep(
            role="assistant",
            content=completion.content,
            metadata={"label": "control_response"},
        )
    ]
    return ScaffoldResult(
        pair_id=pair.id,
        scaffold="control",
        trial=trial,
        answer_x=extract_final_answer(completion.content),
        trace=trace,
    )
