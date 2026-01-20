"""
Agent Orchestrator - Coordinates multiple specialized agents

This module manages the execution and coordination of multiple agents,
determining which agents to use, how to execute them (parallel/sequential),
and how to aggregate their results.
"""

from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_framework import (
    BaseAgent, AgentTask, AgentResult, AgentType,
    AgentRegistry, get_registry
)
from agents import ResearchAgent, CreativeAgent, FactCheckerAgent


class ExecutionStrategy(Enum):
    """Strategy for executing multiple agents."""
    SEQUENTIAL = "sequential"  # One after another
    PARALLEL = "parallel"      # All at once
    PIPELINE = "pipeline"      # Output of one feeds into next


class AgentOrchestrator:
    """
    Orchestrates multiple agents to solve complex tasks.

    The orchestrator determines which agents to use, executes them
    according to a strategy, and aggregates results.
    """

    def __init__(self, registry: Optional[AgentRegistry] = None):
        """
        Initialize the orchestrator.

        Args:
            registry: Agent registry (uses global if not provided)
        """
        self.registry = registry or get_registry()
        self._initialize_default_agents()

    def _initialize_default_agents(self):
        """Register default agents if not already registered."""
        # Check if agents are already registered
        existing_agents = {agent.name for agent in self.registry.get_all_agents()}

        if "ResearchAgent" not in existing_agents:
            self.registry.register(ResearchAgent())

        if "CreativeAgent" not in existing_agents:
            self.registry.register(CreativeAgent())

        if "FactCheckerAgent" not in existing_agents:
            self.registry.register(FactCheckerAgent())

    def execute_multi_agent_task(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        strategy: ExecutionStrategy = ExecutionStrategy.PIPELINE,
        verify_output: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute a task using multiple agents.

        Args:
            query: The user's query
            context: Additional context (user_id, etc.)
            strategy: How to execute agents
            verify_output: Whether to fact-check the results
            progress_callback: Function to call with progress updates

        Returns:
            Dictionary containing aggregated results
        """
        context = context or {}

        # Determine which agents to use
        agent_plan = self._create_agent_plan(query, context)

        if progress_callback:
            progress_callback(f"🤝 **Multi-Agent Mode Activated**\nAgents: {', '.join(agent_plan['agents'])}")

        # Execute based on strategy
        if strategy == ExecutionStrategy.PARALLEL:
            results = self._execute_parallel(query, context, agent_plan, progress_callback)
        elif strategy == ExecutionStrategy.PIPELINE:
            results = self._execute_pipeline(query, context, agent_plan, progress_callback)
        else:  # SEQUENTIAL
            results = self._execute_sequential(query, context, agent_plan, progress_callback)

        # Optionally verify with fact-checker
        if verify_output and "FactCheckerAgent" not in agent_plan["agents"]:
            if progress_callback:
                progress_callback("🔍 **Fact-Checker Agent**: Verifying accuracy...")

            verification = self._verify_results(results)
            results["verification"] = verification

        # Aggregate and format final output
        final_output = self._aggregate_results(results, agent_plan)

        return final_output

    def _create_agent_plan(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine which agents to use and in what order.

        Args:
            query: User query
            context: Task context

        Returns:
            Agent execution plan
        """
        query_lower = query.lower()
        plan = {
            "agents": [],
            "execution_order": [],
            "requires_verification": True
        }

        # Check what capabilities are needed
        needs_research = any(word in query_lower for word in [
            "research", "find", "search", "what", "who", "when", "where",
            "how", "explain", "tell me about", "information"
        ])

        needs_creative = any(word in query_lower for word in [
            "create", "generate", "image", "picture", "story", "write",
            "creative", "imagine", "draw"
        ])

        # Default strategy: use research then creative (pipeline)
        if needs_research:
            plan["agents"].append("ResearchAgent")
            plan["execution_order"].append("ResearchAgent")

        if needs_creative:
            plan["agents"].append("CreativeAgent")
            plan["execution_order"].append("CreativeAgent")

        # If both are needed, creative comes after research (uses research as context)
        # If neither explicitly needed, use both for comprehensive response
        if not plan["agents"]:
            plan["agents"] = ["ResearchAgent", "CreativeAgent"]
            plan["execution_order"] = ["ResearchAgent", "CreativeAgent"]

        return plan

    def _execute_parallel(
        self,
        query: str,
        context: Dict[str, Any],
        plan: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, AgentResult]:
        """
        Execute agents in parallel.

        Args:
            query: User query
            context: Task context
            plan: Agent plan
            progress_callback: Progress callback function

        Returns:
            Dictionary of agent results
        """
        results = {}

        with ThreadPoolExecutor(max_workers=len(plan["agents"])) as executor:
            # Submit all agent tasks
            future_to_agent = {}
            for agent_name in plan["agents"]:
                agent = self.registry.get_agent(agent_name)
                if agent:
                    task = AgentTask(query=query, context=context)
                    future = executor.submit(agent.execute, task)
                    future_to_agent[future] = agent_name

            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    results[agent_name] = result

                    if progress_callback:
                        status = "✓" if result.success else "✗"
                        progress_callback(f"{status} **{agent_name}** completed")

                except Exception as e:
                    print(f"Agent {agent_name} failed: {e}")
                    if progress_callback:
                        progress_callback(f"✗ **{agent_name}** encountered an error")

        return results

    def _execute_sequential(
        self,
        query: str,
        context: Dict[str, Any],
        plan: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, AgentResult]:
        """
        Execute agents one after another.

        Args:
            query: User query
            context: Task context
            plan: Agent plan
            progress_callback: Progress callback function

        Returns:
            Dictionary of agent results
        """
        results = {}

        for agent_name in plan["execution_order"]:
            if progress_callback:
                progress_callback(f"⚙️ **{agent_name}** working...")

            agent = self.registry.get_agent(agent_name)
            if agent:
                task = AgentTask(query=query, context=context)
                result = agent.execute(task)
                results[agent_name] = result

                if progress_callback:
                    status = "✓" if result.success else "✗"
                    progress_callback(f"{status} **{agent_name}** completed")

        return results

    def _execute_pipeline(
        self,
        query: str,
        context: Dict[str, Any],
        plan: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, AgentResult]:
        """
        Execute agents in pipeline (output of one feeds into next).

        Args:
            query: User query
            context: Task context
            plan: Agent plan
            progress_callback: Progress callback function

        Returns:
            Dictionary of agent results
        """
        results = {}
        accumulated_context = context.copy()

        for i, agent_name in enumerate(plan["execution_order"]):
            if progress_callback:
                progress_callback(f"⚙️ **{agent_name}** working...")

            agent = self.registry.get_agent(agent_name)
            if agent:
                task = AgentTask(query=query, context=accumulated_context)
                result = agent.execute(task)
                results[agent_name] = result

                # Pass output to next agent as context
                if result.success and i < len(plan["execution_order"]) - 1:
                    if agent.agent_type == AgentType.RESEARCH:
                        accumulated_context["base_facts"] = result.output
                        accumulated_context["research_sources"] = result.sources
                    elif agent.agent_type == AgentType.CREATIVE:
                        accumulated_context["creative_output"] = result.output

                if progress_callback:
                    status = "✓" if result.success else "✗"
                    progress_callback(f"{status} **{agent_name}** completed")

        return results

    def _verify_results(self, results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """
        Use fact-checker to verify agent results.

        Args:
            results: Results from other agents

        Returns:
            Verification report
        """
        fact_checker = self.registry.get_agent("FactCheckerAgent")
        if not fact_checker:
            return {"verified": False, "reason": "Fact-checker not available"}

        # Combine outputs to verify
        content_to_verify = []
        sources = []

        for agent_name, result in results.items():
            if result.success and result.output:
                if isinstance(result.output, dict):
                    if "text" in result.output:
                        content_to_verify.append(result.output["text"])
                else:
                    content_to_verify.append(str(result.output))

                sources.extend(result.sources)

        if not content_to_verify:
            return {"verified": True, "reason": "No content to verify"}

        combined_content = "\n\n".join(content_to_verify)
        verification = fact_checker.verify_content(combined_content, sources)

        return verification

    def _aggregate_results(
        self,
        results: Dict[str, AgentResult],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple agents into final output.

        Args:
            results: Results from all agents
            plan: Original agent plan

        Returns:
            Aggregated output dictionary
        """
        output = {
            "text": "",
            "image_paths": [],
            "agent_contributions": {},
            "confidence": 0.0,
            "sources": [],
            "verification": None
        }

        # Collect contributions from each agent
        successful_agents = 0
        total_confidence = 0.0

        for agent_name, result in results.items():
            if result.success:
                successful_agents += 1
                total_confidence += result.confidence

                # Add to agent contributions
                output["agent_contributions"][agent_name] = {
                    "type": result.agent_type.value,
                    "confidence": result.confidence,
                    "sources": result.sources
                }

                # Aggregate content
                if isinstance(result.output, dict):
                    if "text" in result.output:
                        output["text"] += f"\n\n**[{agent_name}]**\n{result.output['text']}"
                    if "image_paths" in result.output:
                        output["image_paths"].extend(result.output["image_paths"])
                elif isinstance(result.output, str):
                    output["text"] += f"\n\n**[{agent_name}]**\n{result.output}"

                # Collect sources
                output["sources"].extend(result.sources)

        # Calculate average confidence
        if successful_agents > 0:
            output["confidence"] = total_confidence / successful_agents

        # Add verification if present
        if "verification" in results:
            output["verification"] = results["verification"]

        # Clean up text
        output["text"] = output["text"].strip()

        # Deduplicate sources
        output["sources"] = list(set(output["sources"]))

        return output

    def get_available_agents(self) -> List[str]:
        """Get list of available agent names."""
        return [agent.name for agent in self.registry.get_all_agents()]

    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent info dictionary or None
        """
        agent = self.registry.get_agent(agent_name)
        if agent:
            return {
                "name": agent.name,
                "type": agent.agent_type.value,
                "description": agent.description,
                "status": agent.status.value
            }
        return None


# Global orchestrator instance
_global_orchestrator = None


def get_orchestrator() -> AgentOrchestrator:
    """Get or create the global orchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = AgentOrchestrator()
    return _global_orchestrator
