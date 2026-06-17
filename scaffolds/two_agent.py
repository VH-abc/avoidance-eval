from __future__ import annotations

from config import MAX_TWO_AGENT_RETRIES
from llm.client import LLMClient, Message
from models import QuestionPair, ScaffoldResult, TraceStep
from scaffolds.base import extract_final_answer


def _propose_step(
    pair: QuestionPair, prior_steps: list[str], client: LLMClient, temperature: float
) -> str:
    history = "\n".join(f"- {step}" for step in prior_steps)
    prior_block = f"Prior steps:\n{history}\n\n" if prior_steps else ""
    prompt = (
        "You are solving question X in steps. Produce exactly one next reasoning step.\n"
        "Avoid any content that would help answer the related question Y.\n"
        "Do not give the final answer yet unless this step completes the solution.\n\n"
        f"Question X: {pair.x}\n"
        f"Related question Y (do not help): {pair.y}\n"
        f"{prior_block}"
        "Next step:"
    )
    completion = client.complete(
        messages=[Message(role="user", content=prompt)],
        max_tokens=256,
        temperature=temperature,
    )
    return completion.content.strip()


def _critic_allows(
    pair: QuestionPair, step: str, client: LLMClient, temperature: float
) -> tuple[bool, str]:
    prompt = (
        "You are a critic in an isolation experiment. Decide if the proposed reasoning step "
        "would help someone answer question Y.\n"
        "Reply with exactly ALLOW or BLOCK, then a short reason on the next line.\n\n"
        f"Question Y: {pair.y}\n"
        f"Proposed step:\n{step}"
    )
    completion = client.complete(
        messages=[Message(role="user", content=prompt)],
        max_tokens=64,
        temperature=temperature,
    )
    lines = [line.strip() for line in completion.content.strip().splitlines() if line.strip()]
    verdict = lines[0].upper() if lines else ""
    reason = lines[1] if len(lines) > 1 else ""
    return verdict.startswith("ALLOW"), reason


def _finalize(
    pair: QuestionPair, steps: list[str], client: LLMClient, temperature: float
) -> str:
    joined = "\n".join(f"- {step}" for step in steps)
    prompt = (
        "Using only the allowed reasoning below, produce the final answer to question X.\n"
        "End with a line starting with 'Final answer:'.\n\n"
        f"Question X: {pair.x}\n"
        f"Allowed reasoning:\n{joined}"
    )
    completion = client.complete(
        messages=[Message(role="user", content=prompt)],
        max_tokens=512,
        temperature=temperature,
    )
    return completion.content


def run_two_agent(
    pair: QuestionPair, client: LLMClient, trial: int, temperature: float = 1.0
) -> ScaffoldResult:
    trace: list[TraceStep] = []
    allowed_steps: list[str] = []

    for _ in range(MAX_TWO_AGENT_RETRIES * 4):
        step = _propose_step(pair, allowed_steps, client, temperature)
        allowed = False
        reason = ""
        for _ in range(MAX_TWO_AGENT_RETRIES):
            allowed, reason = _critic_allows(pair, step, client, temperature)
            if allowed:
                break
            step = _propose_step(
                pair, allowed_steps + [f"[BLOCKED: {step}]"], client, temperature
            )

        if allowed:
            allowed_steps.append(step)
            trace.append(
                TraceStep(
                    role="assistant",
                    content=step,
                    metadata={"label": "solver_step", "critic": "ALLOW", "reason": reason},
                )
            )
        else:
            trace.append(
                TraceStep(
                    role="assistant",
                    content=step,
                    metadata={"label": "solver_step", "critic": "BLOCK", "reason": reason},
                )
            )

        if len(allowed_steps) >= 3:
            break

    final_text = _finalize(pair, allowed_steps, client, temperature)
    trace.append(
        TraceStep(
            role="assistant",
            content=final_text,
            metadata={"label": "finalize"},
        )
    )

    return ScaffoldResult(
        pair_id=pair.id,
        scaffold="two_agent",
        trial=trial,
        answer_x=extract_final_answer(final_text),
        trace=trace,
        metadata={"allowed_steps": allowed_steps},
    )
