"""
Research Agent - Specialized in gathering factual information

This agent excels at finding, retrieving, and aggregating information
from multiple sources including web search, documents, and memories.
"""

from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agent_framework import BaseAgent, AgentType, AgentTask, AgentResult, AgentStatus
import mister_fritz


class ResearchAgent(BaseAgent):
    """
    Agent specialized in research and information gathering.

    This agent uses available tools to search multiple sources,
    aggregate findings, and provide comprehensive research results.
    """

    def __init__(
        self,
        name: str = "ResearchAgent",
        model_name: str = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the Research Agent.

        Args:
            name: Agent identifier
            model_name: Name of the LLM model to use
            config: Configuration dictionary
        """
        from fritz_utils import THINKING_OLLAMA_MODEL

        description = (
            "Specializes in gathering factual information from multiple sources. "
            "Uses web search, document retrieval, and memory search to provide "
            "comprehensive, well-sourced research results."
        )

        super().__init__(
            name=name,
            agent_type=AgentType.RESEARCH,
            description=description,
            config=config or {}
        )

        self.model_name = model_name or THINKING_OLLAMA_MODEL
        self.llm = ChatOllama(model=self.model_name, temperature=0.1)

    def can_handle(self, task: AgentTask) -> bool:
        """
        Determine if this agent can handle the task.

        Research agent handles tasks that require:
        - Factual information
        - Web search
        - Document lookup
        - Historical context
        """
        research_keywords = [
            "research", "find", "search", "lookup", "what is",
            "who is", "when", "where", "how", "explain",
            "define", "tell me about", "information about"
        ]

        query_lower = task.query.lower()
        return any(keyword in query_lower for keyword in research_keywords)

    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute research task.

        Args:
            task: The research task to execute

        Returns:
            AgentResult with research findings
        """
        self.status = AgentStatus.WORKING

        try:
            # Determine what research methods to use
            research_plan = self._create_research_plan(task)

            # Execute research
            findings = self._gather_information(task, research_plan)

            # Synthesize results
            synthesis = self._synthesize_findings(task.query, findings)

            self.status = AgentStatus.COMPLETED

            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                success=True,
                output=synthesis,
                confidence=self._calculate_confidence(findings),
                sources=self._extract_sources(findings),
                metadata={
                    "research_plan": research_plan,
                    "findings_count": len(findings),
                    "methods_used": list(research_plan.keys())
                }
            )

        except Exception as e:
            self.status = AgentStatus.FAILED
            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                success=False,
                output=None,
                confidence=0.0,
                error_message=str(e)
            )

    def _create_research_plan(self, task: AgentTask) -> Dict[str, bool]:
        """
        Create a plan for what research methods to use.

        Args:
            task: The task to plan for

        Returns:
            Dictionary of research methods to use
        """
        query_lower = task.query.lower()

        plan = {
            "web_search": False,
            "document_search": False,
            "memory_search": False
        }

        # Check for current/recent information needs
        if any(word in query_lower for word in ["current", "latest", "recent", "today", "news"]):
            plan["web_search"] = True

        # Check for domain-specific knowledge
        if any(word in query_lower for word in ["lore", "document", "file", "record"]):
            plan["document_search"] = True

        # Check for personal/historical context
        if any(word in query_lower for word in ["remember", "told", "said", "previous", "history"]):
            plan["memory_search"] = True

        # Default: use web and document search
        if not any(plan.values()):
            plan["web_search"] = True
            plan["document_search"] = True

        return plan

    def _gather_information(self, task: AgentTask, plan: Dict[str, bool]) -> List[Dict[str, Any]]:
        """
        Gather information using planned methods.

        Args:
            task: The task being executed
            plan: Research plan

        Returns:
            List of findings
        """
        findings = []

        # Web Search
        if plan.get("web_search"):
            try:
                print(f"   [{self.name}] Searching web...")
                web_results = mister_fritz.search_web(task.query)
                if web_results:
                    findings.append({
                        "source": "web_search",
                        "data": web_results,
                        "reliability": 0.7
                    })
            except Exception as e:
                print(f"   [{self.name}] Web search failed: {e}")

        # Document Search
        if plan.get("document_search"):
            try:
                print(f"   [{self.name}] Searching documents...")
                doc_results = mister_fritz.search_documents(task.query)
                if doc_results:
                    findings.append({
                        "source": "document_search",
                        "data": doc_results,
                        "reliability": 0.9
                    })
            except Exception as e:
                print(f"   [{self.name}] Document search failed: {e}")

        # Memory Search
        if plan.get("memory_search") and task.context.get("user_id"):
            try:
                print(f"   [{self.name}] Searching memories...")
                from langchain_core.runnables import RunnableConfig
                config = RunnableConfig(
                    metadata={"user_id": task.context.get("user_id")}
                )
                memory_results = mister_fritz.search_memories_internal(config, task.query)
                if memory_results:
                    findings.append({
                        "source": "memory_search",
                        "data": memory_results,
                        "reliability": 0.8
                    })
            except Exception as e:
                print(f"   [{self.name}] Memory search failed: {e}")

        return findings

    def _synthesize_findings(self, query: str, findings: List[Dict[str, Any]]) -> str:
        """
        Synthesize research findings into a coherent response.

        Args:
            query: Original query
            findings: Gathered research data

        Returns:
            Synthesized response
        """
        if not findings:
            return "I couldn't find sufficient information to answer your query."

        # Prepare context from findings
        context_parts = []
        for finding in findings:
            source = finding["source"]
            data = finding["data"]
            context_parts.append(f"[Source: {source}]\n{data}\n")

        full_context = "\n---\n".join(context_parts)

        # Use LLM to synthesize
        system_prompt = """You are a research analyst synthesizing information from multiple sources.
Your task is to:
1. Combine information from all sources
2. Present facts clearly and concisely
3. Note any conflicting information
4. Cite sources when making claims
5. Be objective and factual"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}\n\nResearch Findings:\n{full_context}\n\nProvide a comprehensive answer:")
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _calculate_confidence(self, findings: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on findings.

        Args:
            findings: Research findings

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not findings:
            return 0.0

        # Weight by reliability scores and count
        total_reliability = sum(f.get("reliability", 0.5) for f in findings)
        avg_reliability = total_reliability / len(findings)

        # Bonus for multiple sources
        source_bonus = min(0.2, len(findings) * 0.05)

        return min(1.0, avg_reliability + source_bonus)

    def _extract_sources(self, findings: List[Dict[str, Any]]) -> List[str]:
        """
        Extract source names from findings.

        Args:
            findings: Research findings

        Returns:
            List of source names
        """
        return [f["source"] for f in findings]
