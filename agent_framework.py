"""
Agent Framework for Multi-Agent Collaboration System

This module provides the base classes and protocols for creating
specialized AI agents that can collaborate to solve complex tasks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import uuid


class AgentType(Enum):
    """Types of specialized agents in the system."""
    RESEARCH = "research"
    CREATIVE = "creative"
    FACT_CHECKER = "fact_checker"
    GENERAL = "general"


class AgentStatus(Enum):
    """Status of an agent's execution."""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentMessage:
    """Message passed between agents or to/from orchestrator."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: __import__('time').time())


@dataclass
class AgentResult:
    """Result from an agent's execution."""
    agent_name: str
    agent_type: AgentType
    success: bool
    output: Any
    confidence: float = 1.0  # 0.0 to 1.0
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class AgentTask:
    """Task assigned to an agent."""
    query: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Dict[str, Any] = field(default_factory=dict)
    tools_allowed: Optional[List[str]] = None
    max_iterations: int = 5
    timeout: float = 60.0  # seconds


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.

    All agents must implement the execute method and can optionally
    override other methods for custom behavior.
    """

    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        description: str,
        tools: Optional[List[Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a base agent.

        Args:
            name: Unique identifier for the agent
            agent_type: Type of agent (research, creative, etc.)
            description: What this agent specializes in
            tools: List of tools/functions the agent can use
            config: Configuration dictionary for agent behavior
        """
        self.name = name
        self.agent_type = agent_type
        self.description = description
        self.tools = tools or []
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.message_history: List[AgentMessage] = []

    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute a task and return results.

        Args:
            task: The task to execute

        Returns:
            AgentResult containing output and metadata
        """
        pass

    def can_handle(self, task: AgentTask) -> bool:
        """
        Determine if this agent can handle a given task.

        Args:
            task: The task to evaluate

        Returns:
            True if the agent can handle the task
        """
        return True  # Default: all agents can attempt any task

    def send_message(self, recipient: str, content: Any, metadata: Optional[Dict] = None) -> AgentMessage:
        """
        Send a message to another agent or the orchestrator.

        Args:
            recipient: Name of the recipient agent
            content: Message content
            metadata: Optional metadata

        Returns:
            The sent message
        """
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            content=content,
            metadata=metadata or {}
        )
        self.message_history.append(message)
        return message

    def receive_message(self, message: AgentMessage):
        """
        Receive and process a message from another agent.

        Args:
            message: The message to receive
        """
        self.message_history.append(message)

    def get_status(self) -> AgentStatus:
        """Get the current status of the agent."""
        return self.status

    def reset(self):
        """Reset the agent to its initial state."""
        self.status = AgentStatus.IDLE
        self.message_history.clear()


class AgentRegistry:
    """
    Registry for managing available agents in the system.
    """

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._agents_by_type: Dict[AgentType, List[BaseAgent]] = {
            agent_type: [] for agent_type in AgentType
        }

    def register(self, agent: BaseAgent):
        """
        Register an agent with the system.

        Args:
            agent: The agent to register
        """
        self._agents[agent.name] = agent
        self._agents_by_type[agent.agent_type].append(agent)

    def unregister(self, agent_name: str):
        """
        Unregister an agent from the system.

        Args:
            agent_name: Name of the agent to remove
        """
        if agent_name in self._agents:
            agent = self._agents[agent_name]
            self._agents_by_type[agent.agent_type].remove(agent)
            del self._agents[agent_name]

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        Get an agent by name.

        Args:
            agent_name: Name of the agent

        Returns:
            The agent or None if not found
        """
        return self._agents.get(agent_name)

    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """
        Get all agents of a specific type.

        Args:
            agent_type: Type of agents to retrieve

        Returns:
            List of agents of that type
        """
        return self._agents_by_type.get(agent_type, [])

    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents."""
        return list(self._agents.values())

    def find_agents_for_task(self, task: AgentTask) -> List[BaseAgent]:
        """
        Find agents capable of handling a task.

        Args:
            task: The task to match agents against

        Returns:
            List of capable agents
        """
        return [agent for agent in self._agents.values() if agent.can_handle(task)]


# Global registry instance
_global_registry = AgentRegistry()


def get_registry() -> AgentRegistry:
    """Get the global agent registry."""
    return _global_registry
