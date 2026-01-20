"""
Fact Checker Agent - Specialized in verifying claims and ensuring accuracy

This agent excels at cross-referencing information, validating claims,
and providing confidence scores for factual statements.
"""

from typing import List, Dict, Any, Tuple
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agent_framework import BaseAgent, AgentType, AgentTask, AgentResult, AgentStatus
import mister_fritz


class FactCheckerAgent(BaseAgent):
    """
    Agent specialized in fact-checking and verification.

    This agent validates claims against multiple sources and provides
    confidence scores for factual accuracy.
    """

    def __init__(
        self,
        name: str = "FactCheckerAgent",
        model_name: str = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the Fact Checker Agent.

        Args:
            name: Agent identifier
            model_name: Name of the LLM model to use
            config: Configuration dictionary
        """
        from fritz_utils import THINKING_OLLAMA_MODEL

        description = (
            "Specializes in fact-checking and verification. "
            "Cross-references claims against multiple sources, identifies "
            "inconsistencies, and provides confidence scores for factual statements."
        )

        super().__init__(
            name=name,
            agent_type=AgentType.FACT_CHECKER,
            description=description,
            config=config or {}
        )

        self.model_name = model_name or THINKING_OLLAMA_MODEL
        # Low temperature for consistent, logical analysis
        self.llm = ChatOllama(model=self.model_name, temperature=0.0)

    def can_handle(self, task: AgentTask) -> bool:
        """
        Determine if this agent can handle the task.

        Fact checker handles tasks that require:
        - Verification of claims
        - Accuracy checking
        - Source validation
        """
        # Fact checker is typically invoked by orchestrator
        # to verify output from other agents
        verification_keywords = [
            "verify", "check", "validate", "confirm",
            "is this true", "is this correct", "fact check"
        ]

        query_lower = task.query.lower()
        return any(keyword in query_lower for keyword in verification_keywords)

    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute fact-checking task.

        Args:
            task: The fact-checking task to execute

        Returns:
            AgentResult with verification results
        """
        self.status = AgentStatus.WORKING

        try:
            # Extract claims to verify
            claims = self._extract_claims(task)

            # Verify each claim
            verification_results = self._verify_claims(claims)

            # Generate verification report
            report = self._generate_verification_report(verification_results)

            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(verification_results)

            self.status = AgentStatus.COMPLETED

            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                success=True,
                output=report,
                confidence=confidence,
                metadata={
                    "claims_checked": len(claims),
                    "verification_results": verification_results,
                    "accuracy_score": confidence
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

    def verify_content(self, content: str, sources: List[str] = None) -> Dict[str, Any]:
        """
        Verify content from other agents.

        Args:
            content: Content to verify
            sources: Optional list of sources used

        Returns:
            Verification result dictionary
        """
        task = AgentTask(
            query=f"Verify this content: {content}",
            context={
                "content": content,
                "sources": sources or []
            }
        )

        result = self.execute(task)
        return {
            "verified": result.success,
            "confidence": result.confidence,
            "report": result.output,
            "issues": result.metadata.get("verification_results", [])
        }

    def _extract_claims(self, task: AgentTask) -> List[str]:
        """
        Extract individual claims from content.

        Args:
            task: The task containing content to analyze

        Returns:
            List of factual claims
        """
        content = task.context.get("content", task.query)

        system_prompt = """You are a claim extraction specialist.
Analyze the provided content and extract distinct factual claims.
Focus on statements that can be verified (avoid opinions or subjective statements).
Return each claim on a new line, numbered."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract factual claims from:\n\n{content}")
        ]

        response = self.llm.invoke(messages)

        # Parse claims from response
        claims = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering/bullets
                claim = line.lstrip('0123456789.-) ').strip()
                if claim:
                    claims.append(claim)

        return claims

    def _verify_claims(self, claims: List[str]) -> List[Dict[str, Any]]:
        """
        Verify each claim against available sources.

        Args:
            claims: List of claims to verify

        Returns:
            List of verification results
        """
        results = []

        for claim in claims:
            print(f"   [{self.name}] Verifying: {claim[:50]}...")
            verification = self._verify_single_claim(claim)
            results.append({
                "claim": claim,
                "status": verification[0],
                "confidence": verification[1],
                "evidence": verification[2]
            })

        return results

    def _verify_single_claim(self, claim: str) -> Tuple[str, float, str]:
        """
        Verify a single claim.

        Args:
            claim: The claim to verify

        Returns:
            Tuple of (status, confidence, evidence)
            status: "verified", "unverified", "contradicted"
        """
        try:
            # Search for evidence
            evidence = []

            # Try web search
            try:
                web_results = mister_fritz.search_web(claim)
                if web_results:
                    evidence.append({
                        "source": "web",
                        "data": web_results
                    })
            except:
                pass

            # Try document search
            try:
                doc_results = mister_fritz.search_documents(claim)
                if doc_results:
                    evidence.append({
                        "source": "documents",
                        "data": doc_results
                    })
            except:
                pass

            if not evidence:
                return ("unverified", 0.3, "No supporting evidence found")

            # Analyze evidence
            return self._analyze_evidence(claim, evidence)

        except Exception as e:
            print(f"   [{self.name}] Verification error: {e}")
            return ("unverified", 0.0, f"Error during verification: {str(e)}")

    def _analyze_evidence(self, claim: str, evidence: List[Dict]) -> Tuple[str, float, str]:
        """
        Analyze evidence to determine claim validity.

        Args:
            claim: The claim being verified
            evidence: Gathered evidence

        Returns:
            Tuple of (status, confidence, explanation)
        """
        # Prepare evidence for analysis
        evidence_text = []
        for item in evidence:
            source = item["source"]
            data = str(item["data"])[:500]  # Limit length
            evidence_text.append(f"[{source}]: {data}")

        full_evidence = "\n---\n".join(evidence_text)

        system_prompt = """You are a fact-checking analyst.
Analyze the evidence and determine if it supports, contradicts, or is neutral to the claim.

Respond in this format:
STATUS: [verified/contradicted/uncertain]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [brief explanation]"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Claim: {claim}\n\nEvidence:\n{full_evidence}")
        ]

        response = self.llm.invoke(messages)

        # Parse response
        status = "unverified"
        confidence = 0.5
        explanation = response.content

        lines = response.content.lower().split('\n')
        for line in lines:
            if line.startswith('status:'):
                if 'verified' in line:
                    status = "verified"
                elif 'contradicted' in line:
                    status = "contradicted"
                elif 'uncertain' in line:
                    status = "uncertain"
            elif line.startswith('confidence:'):
                try:
                    conf_str = line.split(':')[1].strip()
                    confidence = float(conf_str)
                except:
                    pass

        return (status, confidence, explanation)

    def _generate_verification_report(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate a readable verification report.

        Args:
            results: Verification results

        Returns:
            Formatted report string
        """
        report_lines = ["**Fact-Check Report**\n"]

        verified_count = sum(1 for r in results if r["status"] == "verified")
        total_count = len(results)

        report_lines.append(f"Claims Checked: {total_count}")
        report_lines.append(f"Verified: {verified_count}")
        report_lines.append(f"Accuracy: {verified_count/total_count*100:.1f}%\n" if total_count > 0 else "")

        for i, result in enumerate(results, 1):
            status_emoji = {
                "verified": "✓",
                "contradicted": "✗",
                "uncertain": "?",
                "unverified": "?"
            }

            emoji = status_emoji.get(result["status"], "?")
            confidence = result["confidence"]

            report_lines.append(f"{i}. [{emoji}] {result['claim']}")
            report_lines.append(f"   Status: {result['status']} (Confidence: {confidence:.2f})")

            if result["status"] == "contradicted":
                report_lines.append(f"   ⚠️ Warning: This claim may be inaccurate")

            report_lines.append("")

        return "\n".join(report_lines)

    def _calculate_overall_confidence(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate overall confidence in the verified content.

        Args:
            results: Verification results

        Returns:
            Overall confidence score
        """
        if not results:
            return 0.5

        # Weight by individual confidences and statuses
        total_confidence = 0.0
        for result in results:
            status = result["status"]
            confidence = result["confidence"]

            # Penalize contradicted or uncertain claims
            if status == "contradicted":
                total_confidence += (1.0 - confidence) * 0.5
            elif status == "uncertain":
                total_confidence += 0.5
            elif status == "verified":
                total_confidence += confidence
            else:
                total_confidence += 0.3

        return total_confidence / len(results)
