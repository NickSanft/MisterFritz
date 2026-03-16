"""
MisterFritz Skills Directory
=============================
Drop any .py file in here and it will be auto-discovered and loaded on startup.

Convention
----------
Each skill file must expose a `register()` function that returns a dict:

    def register() -> dict[str, tuple[tool, str]]:
        return {
            "tool_name": (tool_function, "Description shown to the agent."),
        }

The tool_function must be a LangChain @tool (or compatible callable).
Files starting with '_' are ignored.

Example skeleton
----------------
    from langchain_core.tools import tool

    @tool(parse_docstring=True)
    def my_tool(arg: str) -> str:
        \"\"\"One-line summary used by the agent.

        Args:
            arg: Description of the argument.

        Returns:
            string: The result.
        \"\"\"
        return f"You passed: {arg}"

    def register() -> dict:
        return {
            "my_tool": (my_tool, "Short description for the system prompt."),
        }
"""
