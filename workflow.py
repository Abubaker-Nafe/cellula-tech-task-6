import json

from evaluator import evaluator_chain
from generator import generator_chain
from agent_memory import AgentMemories
from cache_monitor import cache_stage

MAX_ITERATIONS = 4


def run_workflow(
    question: str,
    context: str,
    memories: AgentMemories | None = None,
) -> dict:
    if memories is None:
        memories = AgentMemories()

    if memories.generator.messages is memories.evaluator.messages:
        raise ValueError(
            "Generator and Evaluator memories must be isolated."
        )

    previous_answer = ""
    feedback = ""
    last_result = {}

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\nIteration {iteration}/{MAX_ITERATIONS}")

        generator_inputs = {
            "question": question,
            "context": context,
            "previous_answer": previous_answer,
            "feedback": feedback,
        }
        with cache_stage("Generator", iteration):
            answer = str(generator_chain.invoke({
                **generator_inputs,
                "generator_history": list(memories.generator.messages),
            }))
        memories.generator.record_turn(generator_inputs, answer)

        evaluator_inputs = {
            "question": question,
            "context": context,
            "answer": answer,
        }
        with cache_stage("Evaluator", iteration):
            evaluation = evaluator_chain.invoke({
                **evaluator_inputs,
                "evaluator_history": list(memories.evaluator.messages),
            })
        memories.evaluator.record_turn(
            evaluator_inputs,
            evaluation.model_dump_json(),
        )

        print("Approved:", evaluation.approved)
        print("Feedback:", evaluation.feedback)
        print(
            "Memory messages:",
            f"Generator={len(memories.generator.messages)},",
            f"Evaluator={len(memories.evaluator.messages)}",
        )

        last_result = {
            "answer": answer,
            "approved": evaluation.approved,
            "feedback": evaluation.feedback,
            "iterations": iteration,
        }

        if evaluation.approved:
            return {**last_result, "status": "approved"}

        previous_answer = answer
        feedback = evaluation.feedback

    return {**last_result, "status": "max_iterations_reached"}

if __name__ == "__main__":
    context = (
        "An LLM cache stores previous model responses. "
        "When the same request is repeated, a matching cached "
        "response can be returned without another LLM call."
    )

    result = run_workflow(
        question="How does caching reduce repeated LLM calls?",
        context=context,
    )

    print("\nFinal result:")
    print(json.dumps(result, indent=2))
