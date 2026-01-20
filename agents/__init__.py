"""
Specialized Agents Package

This package contains specialized agents for the multi-agent collaboration system.
Each agent is designed to handle specific types of tasks.
"""

from .research_agent import ResearchAgent
from .creative_agent import CreativeAgent
from .fact_checker_agent import FactCheckerAgent

__all__ = [
    "ResearchAgent",
    "CreativeAgent",
    "FactCheckerAgent",
]
