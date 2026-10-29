import json
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@dataclass
class AgentMemory:
    messages: list[BaseMessage] = field(default_factory=list)

    def record_turn(self, inputs: dict, output: str) -> None:
        self.messages.extend([
            HumanMessage(
                content=json.dumps(inputs, ensure_ascii=False)
            ),
            AIMessage(content=str(output)),
        ])


@dataclass
class AgentMemories:
    generator: AgentMemory = field(default_factory=AgentMemory)
    evaluator: AgentMemory = field(default_factory=AgentMemory)

    def __post_init__(self) -> None:
        if self.generator.messages is self.evaluator.messages:
            raise ValueError(
                "Generator and Evaluator must have separate memories."
            )