"""
Creative Agent - Specialized in creative content generation

This agent excels at generating engaging narratives, creative responses,
images, and adding personality to interactions.
"""

from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agent_framework import BaseAgent, AgentType, AgentTask, AgentResult, AgentStatus
import mister_fritz


class CreativeAgent(BaseAgent):
    """
    Agent specialized in creative content generation.

    This agent uses higher temperature settings for more creative responses
    and can generate images, write stories, and add personality to answers.
    """

    def __init__(
        self,
        name: str = "CreativeAgent",
        model_name: str = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the Creative Agent.

        Args:
            name: Agent identifier
            model_name: Name of the LLM model to use
            config: Configuration dictionary
        """
        from fritz_utils import THINKING_OLLAMA_MODEL

        description = (
            "Specializes in creative content generation. "
            "Adds engaging narratives, witty responses, storytelling elements, "
            "and can generate images. Uses a butler-like personality."
        )

        super().__init__(
            name=name,
            agent_type=AgentType.CREATIVE,
            description=description,
            config=config or {}
        )

        self.model_name = model_name or THINKING_OLLAMA_MODEL
        # Higher temperature for more creative responses
        self.llm = ChatOllama(model=self.model_name, temperature=0.8)

    def can_handle(self, task: AgentTask) -> bool:
        """
        Determine if this agent can handle the task.

        Creative agent handles tasks that require:
        - Image generation
        - Storytelling
        - Creative writing
        - Humor
        - Personality
        """
        creative_keywords = [
            "create", "generate", "image", "picture", "draw",
            "story", "write", "imagine", "creative", "funny",
            "joke", "entertain", "describe creatively"
        ]

        query_lower = task.query.lower()
        return any(keyword in query_lower for keyword in creative_keywords)

    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute creative task.

        Args:
            task: The creative task to execute

        Returns:
            AgentResult with creative output
        """
        self.status = AgentStatus.WORKING

        try:
            # Determine what creative elements are needed
            creative_plan = self._create_creative_plan(task)

            # Generate creative content
            output = self._generate_creative_content(task, creative_plan)

            self.status = AgentStatus.COMPLETED

            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                success=True,
                output=output,
                confidence=0.9,  # Creative content is subjective
                metadata={
                    "creative_plan": creative_plan,
                    "has_image": creative_plan.get("generate_image", False),
                    "style": creative_plan.get("style", "witty")
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

    def _create_creative_plan(self, task: AgentTask) -> Dict[str, Any]:
        """
        Create a plan for creative content generation.

        Args:
            task: The task to plan for

        Returns:
            Dictionary describing creative approach
        """
        query_lower = task.query.lower()

        plan = {
            "generate_image": False,
            "add_storytelling": False,
            "add_humor": False,
            "style": "witty"
        }

        # Check for image generation needs
        if any(word in query_lower for word in ["image", "picture", "draw", "generate", "create image"]):
            plan["generate_image"] = True

        # Check for storytelling needs
        if any(word in query_lower for word in ["story", "tale", "narrative", "describe"]):
            plan["add_storytelling"] = True

        # Check for humor needs
        if any(word in query_lower for word in ["funny", "joke", "humor", "entertain"]):
            plan["add_humor"] = True
            plan["style"] = "humorous"

        # Check for factual content that needs creative presentation
        if task.context.get("base_facts"):
            plan["add_storytelling"] = True

        return plan

    def _generate_creative_content(self, task: AgentTask, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate creative content based on plan.

        Args:
            task: The task being executed
            plan: Creative plan

        Returns:
            Dictionary with generated content
        """
        output = {
            "text": "",
            "image_paths": []
        }

        # Generate image if requested
        if plan.get("generate_image"):
            image_path = self._generate_image(task)
            if image_path:
                output["image_paths"].append(image_path)

        # Generate creative text
        output["text"] = self._generate_creative_text(task, plan)

        return output

    def _generate_image(self, task: AgentTask) -> str:
        """
        Generate an image based on task query.

        Args:
            task: The task with image generation request

        Returns:
            Path to generated image or empty string
        """
        try:
            print(f"   [{self.name}] Generating image...")

            # Extract image prompt from query
            prompt = task.query

            # Try to find prompt after certain keywords
            for keyword in ["image of", "picture of", "draw", "generate", "create"]:
                if keyword in prompt.lower():
                    parts = prompt.lower().split(keyword)
                    if len(parts) > 1:
                        prompt = parts[1].strip()
                        break

            image_path = mister_fritz.generate_image(prompt)
            print(f"   [{self.name}] Image generated: {image_path}")
            return image_path

        except Exception as e:
            print(f"   [{self.name}] Image generation failed: {e}")
            return ""

    def _generate_creative_text(self, task: AgentTask, plan: Dict[str, Any]) -> str:
        """
        Generate creative text response.

        Args:
            task: The task being executed
            plan: Creative plan

        Returns:
            Creative text response
        """
        # Build system prompt based on creative style
        style = plan.get("style", "witty")

        system_prompts = {
            "witty": """You are Mister Fritz, a sophisticated AI with the personality of an English butler.
Respond with wit, charm, and a touch of sarcasm. Be helpful but entertaining.
Keep responses engaging and memorable.""",

            "humorous": """You are Mister Fritz, an AI comedian with the refinement of an English butler.
Make the user laugh while still being helpful. Use clever wordplay and dry humor.
Be entertaining above all else.""",

            "storytelling": """You are Mister Fritz, a masterful storyteller with butler-like sophistication.
Weave information into engaging narratives. Make facts memorable through stories.
Be vivid and descriptive in your language."""
        }

        system_prompt = system_prompts.get(style, system_prompts["witty"])

        # Add context if available
        context = ""
        if task.context.get("base_facts"):
            context = f"\n\nBase Information:\n{task.context['base_facts']}\n\n"
            system_prompt += f"\n\nUse the base information provided and present it in an engaging, creative way."

        # Generate response
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{context}User Query: {task.query}\n\nProvide a creative, engaging response:")
        ]

        response = self.llm.invoke(messages)
        return response.content

    def add_personality_to_facts(self, facts: str, style: str = "witty") -> str:
        """
        Transform dry facts into engaging content.

        Args:
            facts: Raw factual information
            style: Personality style to apply

        Returns:
            Facts presented with personality
        """
        system_prompt = f"""You are Mister Fritz, transforming dry facts into engaging content.
Maintain accuracy but add personality and charm.
Style: {style}
Keep the core information accurate but make it memorable."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Transform this information:\n\n{facts}")
        ]

        response = self.llm.invoke(messages)
        return response.content
