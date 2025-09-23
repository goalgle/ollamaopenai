from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

try:
    from agents import Agent, Runner
except ImportError:
    # Fallback for testing without agents package
    class Agent:
        def __init__(self, name, instructions, model, **kwargs):
            self.name = name
            self.instructions = instructions
            self.model = model
            # Accept all keyword arguments for compatibility
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Runner:
        @staticmethod
        def run_sync(agent, message):
            # Mock response for testing
            class MockResult:
                def __init__(self):
                    self.final_output = f"Mock response from {agent.name}: {message}"
            return MockResult()

@dataclass
class ConversationResult:
    agent_name: str
    user_input: str
    final_output: str
    execution_time: float
    success: bool
    timestamp: datetime
    error: Optional[str] = None

class ChatManager:
    """Manages interactive conversations with multiple agents"""

    def __init__(self, agents: Dict[str, Agent], default_agent: str = "math"):
        self.agents = agents
        self.current_agent_name = default_agent
        self.conversation_history: List[ConversationResult] = []

        if default_agent not in agents:
            raise ValueError(f"Default agent '{default_agent}' not found in available agents")

    @property
    def current_agent(self) -> Agent:
        return self.agents[self.current_agent_name]

    def switch_agent(self, agent_name: str) -> bool:
        """Switch to different agent"""
        if agent_name in self.agents:
            self.current_agent_name = agent_name
            return True
        return False

    def process_message(
        self,
        message: str,
        agent_name: Optional[str] = None
    ) -> ConversationResult:
        """Process message with specified or current agent"""

        target_agent_name = agent_name or self.current_agent_name
        target_agent = self.agents[target_agent_name]

        start_time = datetime.now()

        try:
            result = Runner.run_sync(target_agent, message)
            end_time = datetime.now()

            execution_time = (end_time - start_time).total_seconds()

            conversation_result = ConversationResult(
                agent_name=target_agent_name,
                user_input=message,
                final_output=result.final_output,
                execution_time=execution_time,
                success=True,
                timestamp=start_time
            )

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            conversation_result = ConversationResult(
                agent_name=target_agent_name,
                user_input=message,
                final_output="",
                execution_time=execution_time,
                success=False,
                timestamp=start_time,
                error=str(e)
            )

        self.conversation_history.append(conversation_result)
        return conversation_result