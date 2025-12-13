import argparse
import re
import sys

from duckduckgo_search import DDGS
import requests
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from typing import Any, Dict, Iterable, List

# LangChain imports
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    BaseMessage,
)

from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.runnables import RunnableLambda


def _coerce_text(content: Any) -> str:
    """Best-effort conversion of message content (can be str or structured) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Some models may return a list of content parts (e.g., [{'type':'text','text':'...'}])
    if isinstance(content, list):
        parts: List[str] = []
        for c in content:
            if isinstance(c, dict) and "text" in c:
                parts.append(str(c.get("text", "")))
            else:
                parts.append(str(c))
        return "".join(parts)
    return str(content)


def build_graph(llm: Any):
    """Create a minimal LangGraph with a single LLM node."""
    graph = StateGraph(MessagesState)
    # Wrap the model so the node receives just the messages and returns state-shaped output
    # State -> List[BaseMessage] -> AIMessage/AIMessageChunk -> {"messages": [AIMessage/...]}.
    node = (
        RunnableLambda(lambda state: state["messages"])  # extract messages list from state
        | llm
        | RunnableLambda(lambda msg: {"messages": [msg]})  # wrap back into state shape
    )

    graph.add_node("model", node)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile()


def stream_chat(app, prompt: str) -> None:
    """Stream the model output token-by-token using LangGraph's streaming API."""
    # We send the initial human message into the graph's state
    initial_state: Dict[str, List[BaseMessage]] = {
        "messages": [HumanMessage(content=prompt)]
    }

    # Print a simple prefix to indicate assistant output
    sys.stdout.write("Assistant: ")
    sys.stdout.flush()

    # Stream incremental state updates from the graph. When the underlying LLM
    # supports token streaming, this yields AIMessageChunk objects as they arrive.
    for values in app.stream(initial_state, stream_mode="values"):
        msgs: List[BaseMessage] = values.get("messages", [])  # type: ignore[assignment]
        for m in msgs:
            if isinstance(m, AIMessageChunk):
                sys.stdout.write(_coerce_text(m.content))
                sys.stdout.flush()
            elif isinstance(m, AIMessage):
                # Some models don't stream chunks; they return a final AIMessage only.
                sys.stdout.write(_coerce_text(m.content))
                sys.stdout.flush()

    # Ensure final newline after streaming finishes
    sys.stdout.write("\n")
    sys.stdout.flush()

@tool(parse_docstring=True)
def scrape_web(url: str):
    """
    Takes in the string of the URL and returns results all readable text from the website

    Args:
    url: The URL to pull from the internet.

    Returns:
    string: The readable text from the website.
    """
    try:
        # Import locally to avoid a hard dependency if the tool isn't used
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError:
            return "Error: BeautifulSoup (bs4) is not installed. Please install it with 'pip install beautifulsoup4'."

        # 1. Fetch the content
        response = requests.get(url)
        response.raise_for_status()

        # 2. Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 3. Remove non-visible tags
        # We explicitly remove script, style, metadata, title, and comments
        for element in soup(['script', 'style', 'head', 'title', 'meta', '[document]']):
            element.extract()

        # 4. Extract text
        # get_text() with separator=' ' prevents words merging (e.g. "hello</div><div>world")
        text = soup.get_text(separator=' ')

        # 5. Clean up whitespace
        # This collapses multiple spaces/newlines into a single space and strips edges
        clean_text = re.sub(r'\s+', ' ', text).strip()

        return clean_text
    except Exception as e:
        return f"Error: {e}"

@tool(parse_docstring=True)
def search_web(text_to_search: str):
    """
    Takes in a string and returns results from the internet.

    Args:
    text_to_search: The text to search the internet for information.

    Returns:
    list: A list of dictionaries, each containing string keys and string values representing the search results.
    """
    results = DDGS().text(text_to_search, max_results=5)
    print(results)
    return results

def get_conversation_tools_description():
    """
    Returns a dictionary of available tools and their descriptions.
    """
    conversation_tool_dict = {
        "scrape_website": (scrape_web, "If provided a URL by the user, this can be used to scrape a website's HTML."),
        "search_web": (search_web, "Use only to search the internet if you are unsure about something.")
    }
    return conversation_tool_dict


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Simple LangChain + LangGraph CLI using Ollama.")
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The prompt to send to the model. If omitted, you will be asked to type it interactively.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model name to use (default: llama3.2)",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Base URL for the local Ollama server (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )

    args = parser.parse_args(argv)

    prompt = args.prompt
    if not prompt:
        try:
            prompt = input("You: ")
        except KeyboardInterrupt:
            return 130

    # Instantiate the Ollama-backed chat model
    llm = ChatOllama(model=args.model, base_url=args.base_url, temperature=args.temperature)

    # Build and run a one-node LangGraph that calls the model
    app = build_graph(llm)

    try:
        stream_chat(app, prompt)
    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C while streaming
        sys.stdout.write("\n[Interrupted]\n")
        sys.stdout.flush()
        return 130
    except Exception as e:  # Provide a helpful error if Ollama/model isn't available
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(
            "Make sure Ollama is running (ollama serve) and the model '"
            + args.model
            + "' is installed (e.g., 'ollama pull "
            + args.model
            + "').\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
