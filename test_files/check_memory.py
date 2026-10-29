from agent_memory import AgentMemories

memories = AgentMemories()

question = "What does caching do?"
context = "Caching reuses stored responses."
answer = "It reuses stored responses."

print(
    "Shared memory object:",
    memories.generator is memories.evaluator,
)
print(
    "Shared messages list:",
    memories.generator.messages is memories.evaluator.messages,
)

memories.generator.record_turn(
    {"question": question, "context": context},
    answer,
)

print(
    "After Generator -> Generator:",
    len(memories.generator.messages),
    "Evaluator:",
    len(memories.evaluator.messages),
)

memories.evaluator.record_turn(
    {"question": question, "context": context, "answer": answer},
    '{"approved": true, "feedback": "Supported by the context."}',
)

print(
    "After Evaluator -> Generator:",
    len(memories.generator.messages),
    "Evaluator:",
    len(memories.evaluator.messages),
)