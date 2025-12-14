# ===== IMPORTS =====
import json
import random
import re
import uuid
from contextlib import ExitStack
from datetime import datetime
from typing import Literal
import requests
from bs4 import BeautifulSoup

import pytz
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, BaseTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langchain.agents import create_agent

from chroma_store import ChromaStore
# ===== LOCAL MODULES =====
from message_source import MessageSource
from sqlite_store import SQLiteStore

# ===== CONFIGURATION =====
OLLAMA_MODEL = "gpt-oss"
DB_NAME = "chat_history.db"

# Constants for the routing decisions
CONVERSATION_NODE = "conversation"
SUMMARIZE_CONVERSATION_NODE = "summarize_conversation"


def get_conversation_tools_description():
    """
    Returns a dictionary of available tools and their descriptions.
    """
    conversation_tool_dict = {
        "get_current_time": (get_current_time, "Fetch the current time (US / Central Standard Time)."),
        "scrape_website": (scrape_web, "If provided a URL by the user, this can be used to scrape a website's HTML."),
        "search_web": (search_web, "Use only to search the internet if you are unsure about something."),
        "roll_dice": (roll_dice, "Roll different types of dice."),
        "search_memories": (search_memories, "Returns a JSON payload of stored memories you have had with a user based on a search term.")
    }
    return conversation_tool_dict

# ===== UTILITY FUNCTIONS =====
def get_system_description(tools: dict[str, tuple[BaseTool, str]]):
    """
    Format the chatbot's system role description dynamically by including tools from the list.
    """
    # Set tools list dynamically
    tool_descriptions = "".join(
        [f"    {tool_name}: {tup[1]}\n" for tool_name, tup in tools.items()]
    )

    return f"""
Role:
    You are an AI conversationalist named Mister Fritz, you respond to the user's messages with sophisticated, sardonic, and witty remarks like an English butler.
    You do retain memories per user, and can use the search_memories tool to retrieve them. Always search for memories first before searching the web.
    When responding to the user, keep your response to a paragraph or less.

Tools:
{tool_descriptions}
    """


def get_source_info(source: MessageSource, user_id: str) -> str:
    """Generate source information based on the messaging platform."""
    if source == MessageSource.DISCORD_TEXT:
        return f"User is texting from Discord (User ID: {user_id})"
    elif source == MessageSource.DISCORD_VOICE:
        return f"User is speaking from Discord (User ID: {user_id}). Answer in 10 words or less."
    return f"User is interacting via CLI (User ID: {user_id})"


def format_prompt(prompt: str, source: MessageSource, user_id: str) -> str:
    """Format the final prompt for the chatbot."""
    return f"""
    Context:
        {get_source_info(source, user_id)}
    Question:
        {prompt}
    """


def search_memories_internal(config: RunnableConfig, query: str):
    user_id = config.get("metadata").get("user_id")
    search_result = chroma_store.search(query, (str(user_id),), limit=30)
    summaries = {}
    for _, summary_dict in search_result:
        for key, summary in summary_dict.items():
            summaries[key] = summary
    json_summaries = json.dumps(summaries)
    print(json_summaries)
    return json_summaries


@tool(parse_docstring=True)
def get_current_time():
    """
    Returns the current time as a string in RFC3339 (YYYY-MM-DDTHH:MM:SS) format.

    Example - 2025-01-13T23:11:56.337644-06:00
    """
    return get_current_time_internal()


def get_current_time_internal():
    # Get the current time in UTC
    utc_now = datetime.now(pytz.utc)

    # Convert to CST (Central Standard Time)
    cst = pytz.timezone('US/Central')
    cst_now = utc_now.astimezone(cst)

    # Format the timestamp in RFC3339 format
    rfc3339_timestamp = cst_now.isoformat()

    print(rfc3339_timestamp)
    return rfc3339_timestamp

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
        text = soup.get_text(' ')

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


@tool(parse_docstring=True)
def roll_dice(num_dice: int, num_sides: int, config: RunnableConfig):
    """
    Rolls a specified number of dice, each with a specified number of sides.

    Args:
    num_dice: The number of dice to roll.
    num_sides: The number of sides on each die.
    config: The RunnableConfig.

    Returns:
    list: A list containing the result of each die roll.
    """
    user_id = config.get("metadata").get("user_id")
    if num_dice <= 0 or num_sides <= 0:
        raise ValueError("Both number of dice and number of sides must be positive integers.")

    rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
    return (f"Here are the results: {user_id}."
            f" {rolls}")


@tool(parse_docstring=True)
def search_memories(config: RunnableConfig, query: str):
    """ This function returns memories in JSON format based on a search term.

    Args:
        config: The RunnableConfig.
        query: The keywords do to a semantic search for.
    """
    print("TOOL CALLED")
    return search_memories_internal(config, query)


def add_memory(user_id: str, memory_key: str, memory_to_store: str):
    """ This function stores a memory. Only use this if the user has asked you to.

    Args:
        user_id (str): The id of the user to store the memory.
        memory_key (str): A unique identifier for the memory.
        memory_to_store (str): The memory you wish to store.
    """
    memory_dict = {memory_key: memory_to_store}
    chroma_store.put((user_id,), str(uuid.uuid4()), memory_dict)
    return "Added memory for {}: {}".format(memory_key, memory_to_store)


# ===== MAIN FUNCTION =====
def ask_stuff(base_prompt: str, source: MessageSource, user_id: str) -> str:
    """Process user input and return the chatbot's response."""
    user_id_clean = re.sub(r'[^a-zA-Z0-9]', '', user_id)  # Clean special characters
    full_prompt = format_prompt(base_prompt, source, user_id_clean)

    system_prompt = get_system_description(get_conversation_tools_description())
    print(f"Role description: {system_prompt}")
    print(f"Prompt to ask: {full_prompt}")

    config = {"configurable": {"user_id": user_id_clean, "thread_id": user_id_clean}}
    inputs = {"messages": [("user", full_prompt)]}

    return print_stream(app.stream(inputs, config=config, stream_mode="values"))


def print_stream(stream):
    """Process and print streamed messages."""
    message = ""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
    return message.content


# ===== SETUP & INITIALIZATION =====
conversation_tools = [tool_info[0] for tool_info in get_conversation_tools_description().values()]
print(conversation_tools)

store = SQLiteStore(DB_NAME)
chroma_store = ChromaStore()
exit_stack = ExitStack()
checkpointer = exit_stack.enter_context(SqliteSaver.from_conn_string(DB_NAME))
ollama_instance = ChatOllama(model=OLLAMA_MODEL)

conversation_react_agent = create_agent(ollama_instance, tools=conversation_tools)


def should_continue(state: MessagesState) -> Literal["summarize_conversation", "__end__"]:
    #messages = state["messages"]
    #print(f"Messages: {messages}")
    """Decide whether to summarize or end the conversation."""
    return SUMMARIZE_CONVERSATION_NODE if len(state["messages"]) > 15 else END


def summarize_conversation(state: MessagesState, config: RunnableConfig):
    print("In: summarize_conversation")
    metadata = config.get("metadata", {})
    user_id = metadata.get("user_id")
    summary_message_prompt = "Please summarize the conversation above:"
    messages = state["messages"]
    # messages[-1].content = messages[-1].content + "\r\n I am wrapping up this conversation and starting a new one :)"
    messages = messages + [HumanMessage(content=summary_message_prompt)]
    summary_response = ollama_instance.invoke(messages)
    timestamp = get_current_time_internal()
    summary = f"Summary made at {timestamp} \r\n {summary_response.content}"
    print(f"Summary: {summary}")
    response_key_inputs = [
        ("system",
         "Please provide a short sentence describing this memory starting with the word \"memory\". Example - memory_of_pie"),
        ("user", summary)]
    summary_response_key = ollama_instance.invoke(response_key_inputs, config=get_config_values(config))
    print(f"Summary Key: {summary_response_key.content}")
    add_memory(user_id, summary_response_key.content, summary)
    # Remove all but the last message
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-1]]

    return {"messages": delete_messages}


def conversation(state: MessagesState, config: RunnableConfig):
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    print(f"Latest messsage: {latest_message}")
    inputs = {"messages": [("system", get_system_description(get_conversation_tools_description())),
                           ("user", latest_message)]}
    resp = print_stream(conversation_react_agent.stream(inputs, config=get_config_values(config), stream_mode="values"))
    return {'messages': [resp]}


def get_config_values(config: RunnableConfig) -> RunnableConfig:
    metadata = config.get("metadata", {})
    config_values: RunnableConfig = {
        "configurable": {
            "user_id": metadata.get("user_id"),
            "thread_id": metadata.get("thread_id"),
        }
    }
    return config_values


# ===== GRAPH WORKFLOW =====
workflow = StateGraph(MessagesState)

# Define nodes
workflow.add_node(CONVERSATION_NODE, conversation)
workflow.add_node(SUMMARIZE_CONVERSATION_NODE, summarize_conversation)

# Set workflow edges
workflow.add_edge(START, CONVERSATION_NODE)
workflow.add_conditional_edges(CONVERSATION_NODE, should_continue)
workflow.add_edge(SUMMARIZE_CONVERSATION_NODE, END)

# Compile graph
app = workflow.compile(checkpointer=checkpointer, store=store)


# with open("mermaid_diagram.png", "wb") as binary_file:
#     binary_file.write(app.get_graph().draw_mermaid_png())