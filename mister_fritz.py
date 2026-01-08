import json
import random
import re
import uuid
from contextlib import ExitStack
from datetime import datetime
from typing import Literal, TypedDict, Annotated
import requests
from bs4 import BeautifulSoup

import pytz
from ddgs import DDGS
from langchain_core.messages import HumanMessage, RemoveMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, BaseTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages
from langchain.agents import create_agent

import document_engine
import image_generator
from chroma_store import ChromaStore
from fritz_utils import MessageSource, DOC_STORAGE_DESCRIPTION, CHAT_DB_NAME, THINKING_OLLAMA_MODEL, VISION_MODEL
from sqlite_store import SQLiteStore

CONVERSATION_NODE = "conversation"
SUMMARIZE_CONVERSATION_NODE = "summarize_conversation"


# Define extended state for tracking attachments
class EnhancedState(TypedDict):
    messages: Annotated[list, add_messages]
    image_paths: list[str]  # For generated images (output)
    user_image_paths: list[str]  # For user-provided images (input)


def get_conversation_tools_description():
    """
    Returns a dictionary of available tools and their descriptions.
    """
    conversation_tool_dict = {
        "get_current_time": (get_current_time, "Fetch the current time (US / Central Standard Time)."),
        "scrape_web": (scrape_web, "If provided a URL by the user, this can be used to scrape a website's HTML."),
        "search_web": (search_web, "Use only to search the internet if you are unsure about something."),
        "roll_dice": (roll_dice, "Roll different types of dice."),
        "search_memories": (search_memories, "Returns a JSON payload of stored memories you have had with a user based on a search term."),
        "search_documents": (search_documents, f"Search local documents. Use this for questions about: {DOC_STORAGE_DESCRIPTION}"),
        "generate_image": (generate_image, "Generates an image based on a given prompt."),
        "analyze_image": (analyze_image, "Analyzes an image. If the user asks about an image, assume that the tool knows its location.")
    }
    return conversation_tool_dict

# ===== UTILITY FUNCTIONS =====
def get_system_description(tools: dict[str, tuple[BaseTool, str]]):
    """
    Format the chatbot's system role description dynamically by including tools from the list.
    """
    tool_descriptions = "".join(
        [f"    {tool_name}: {tup[1]}\n" for tool_name, tup in tools.items()]
    )

    return f"""
Role:
    You are an AI conversationalist named Mister Fritz, you respond to the user's messages with sophisticated, sardonic, and witty remarks like an English butler.
    You do retain memories per user, and can use the search_memories tool to retrieve them when relevant to the conversation.

Tools:
{tool_descriptions}
    """


def get_source_info(source: MessageSource, user_id: str) -> str:
    """Generate source information based on the messaging platform."""
    if source == MessageSource.DISCORD_TEXT:
        return f"User is texting from Discord (User ID: {user_id})"
    elif source == MessageSource.DISCORD_TEXT_AND_IMAGE:
        return f"User is texting from Discord with and has an image that the analyze_image tool has the path for already. (User ID: {user_id})"
    elif source == MessageSource.DISCORD_VOICE:
        return f"User is speaking from Discord (User ID: {user_id}). Please answer in 30 words or less."
    return f"User is interacting via CLI (User ID: {user_id})"


def format_prompt(prompt: str, source: MessageSource, user_id: str, additional_info="") -> str:
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
def generate_image(prompt: str):
    """
    Use this tool to generate an image if the user asks you to

    Args:
    prompt: The prompt to give to Stable Diffusion to generate the image.

    Returns:
    string: The path of the image.
    """
    return image_generator.generate_image(prompt)



@tool(parse_docstring=True)
def search_documents(query: str):
    """
    Use this tool to get information when the user asks about the description provided in the prompt.

    Args:
    query: The question or search term to look for in the documents that match the search_documents description.

    Returns:
    string: The answer derived from the documents.
    """
    return document_engine.query_documents(query)


@tool(parse_docstring=True)
def search_memories(config: RunnableConfig, query: str):
    """ This function returns memories in JSON format based on a search term.

    Args:
        config: The RunnableConfig.
        query: The keywords do to a semantic search for.
    """
    return search_memories_internal(config, query)


@tool(parse_docstring=True)
def analyze_image(config: RunnableConfig, question: str = "What is in this image?"):
    """
    Analyzes images using a vision model which has already been downloaded.

    Args:
        config: The RunnableConfig containing user_image_paths in metadata.
        question: The specific question to ask about the image(s). Default: "What is in this image?"

    Returns:
        string: A description of what's in the image(s).
    """
    import base64
    import ollama

    # Get user images from config
    metadata = config.get("metadata", {})
    user_images = metadata.get("user_image_paths", [])
    print(f"User images: {user_images}")

    if not user_images:
        return "No images were attached to analyze. Please ask the user to attach an image."

    try:
        # Prepare images for vision model (read and encode)
        encoded_images = []
        for img_path in user_images:
            try:
                with open(img_path, 'rb') as image_file:
                    encoded_images.append(base64.b64encode(image_file.read()).decode('utf-8'))
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")

        if not encoded_images:
            return "Could not read the attached images. Please try again."

        # Call vision model
        print(f"Analyzing {len(encoded_images)} image(s) with {VISION_MODEL}")
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': question,
                'images': encoded_images
            }]
        )

        analysis = response['message']['content']
        print(f"Vision analysis complete: {len(analysis)} chars")
        return analysis

    except Exception as e:
        error_msg = f"Error analyzing image: {str(e)}"
        print(error_msg)
        return error_msg


def add_memory(user_id: str, memory_key: str, memory_to_store: str):
    """ This function stores a memory. Only use this if the user has asked you to.

    Args:
        user_id (str): The id of the user to store the memory.
        memory_key (str): A unique identifier for the memory.
        memory_to_store (str): The memory you wish to store.
    """
    memory_dict = {memory_key: memory_to_store}
    chroma_store.put((str(user_id),), str(uuid.uuid4()), memory_dict)
    return "Added memory for {}: {}".format(memory_key, memory_to_store)


# ===== MAIN FUNCTION =====
def ask_stuff(base_prompt: str, source: MessageSource, user_id: str, progress_callback=None, streaming_callback=None, user_image_paths: list[str] = None) -> dict:
    """Process user input and return structured output with text and attachments.

    Args:
        base_prompt: The user's message/question
        source: The messaging platform source
        user_id: The user's identifier
        progress_callback: Optional function to call with progress updates.
                          Should accept a single string argument.
        streaming_callback: Optional function to call with partial text as it streams.
                           Should accept a single string argument with accumulated text.
        user_image_paths: Optional list of file paths to images the user has attached.
    """
    user_id_clean = re.sub(r'[^a-zA-Z0-9]', '', user_id)  # Clean special characters
    # Add user images to metadata if provided
    if user_image_paths:
        full_prompt = format_prompt(base_prompt, source, user_id_clean, f" User has attached images: {user_image_paths}")
    else:
        user_image_paths = []
        full_prompt = format_prompt(base_prompt, source, user_id_clean)

    system_prompt = get_system_description(get_conversation_tools_description())
    print(f"Role description: {system_prompt}")
    print(f"Prompt to ask: {full_prompt}")



    config = {
        "configurable": {"user_id": user_id_clean, "thread_id": user_id_clean},
        "metadata": {
            "user_id": user_id_clean,
            "thread_id": user_id_clean,
            "progress_callback": progress_callback,
            "streaming_callback": streaming_callback,
            "user_image_paths": user_image_paths
        }
    }
    inputs = {"messages": [("user", full_prompt)], "image_paths": [], "user_image_paths": user_image_paths}

    # Collect final state from stream
    final_state = None
    for s in app.stream(inputs, config=config, stream_mode="values"):
        final_state = s
        # Still print messages for backwards compatibility
        message = s["messages"][-1] if "messages" in s and s["messages"] else None
        if message:
            if isinstance(message, tuple):
                print(message)
            elif hasattr(message, 'pretty_print'):
                message.pretty_print()

    # Extract structured output
    final_text = ""
    if final_state and "messages" in final_state and final_state["messages"]:
        last_msg = final_state["messages"][-1]
        final_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

    image_paths = final_state.get("image_paths", []) if final_state else []

    return {
        "text": final_text,
        "image_paths": image_paths,
        "timestamp": get_current_time_internal()
    }


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

# Cache system prompt at initialization (one-time cost)
CACHED_SYSTEM_PROMPT = get_system_description(get_conversation_tools_description())

store = SQLiteStore(CHAT_DB_NAME)
chroma_store = ChromaStore()
exit_stack = ExitStack()
checkpointer = exit_stack.enter_context(SqliteSaver.from_conn_string(CHAT_DB_NAME))
ollama_instance = ChatOllama(model=THINKING_OLLAMA_MODEL)

conversation_react_agent = create_agent(ollama_instance, tools=conversation_tools)


def should_continue(state: EnhancedState) -> Literal["summarize_conversation", "__end__"]:
    #messages = state["messages"]
    #print(f"Messages: {messages}")
    """Decide whether to summarize or end the conversation."""
    return SUMMARIZE_CONVERSATION_NODE if len(state["messages"]) > 15 else END


def summarize_conversation(state: EnhancedState, config: RunnableConfig):
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


def conversation(state: EnhancedState, config: RunnableConfig):
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    print(f"Latest message: {latest_message}")
    # Use cached system prompt instead of rebuilding
    inputs = {"messages": [("system", CACHED_SYSTEM_PROMPT),
                           ("user", latest_message)]}

    # Get progress and streaming callbacks from config if available
    metadata = config.get("metadata", {})
    progress_callback = metadata.get("progress_callback")
    streaming_callback = metadata.get("streaming_callback")
    print(f"Progress callback available: {progress_callback is not None}")
    print(f"Streaming callback available: {streaming_callback is not None}")

    # Tool name to user-friendly message mapping
    tool_messages = {
        "generate_image": "Generating an image, this may take a moment...",
        "search_documents": "Searching through documents for you...",
        "search_web": "Searching the web...",
        "scrape_website": "Scraping website content...",
        "search_memories": "Looking through my memories...",
        "analyze_image": "🔍 Analyzing your image(s) with vision AI...",
    }

    # Track which tools we've already notified about
    notified_tools = set()

    # Stream and collect the response
    final_state = None
    accumulated_text = ""

    for s in conversation_react_agent.stream(inputs, config=get_config_values(config), stream_mode="values"):
        final_state = s

        # Check for tool calls in the latest message
        if "messages" in s and s["messages"]:
            latest = s["messages"][-1]
            # Check if this is an AI message with tool calls
            if hasattr(latest, 'tool_calls') and latest.tool_calls:
                print(f"Detected tool calls: {[tc.get('name', '') for tc in latest.tool_calls]}")
                if progress_callback:
                    for tool_call in latest.tool_calls:
                        tool_name = tool_call.get('name', '')
                        # Only notify once per tool type
                        if tool_name in tool_messages and tool_name not in notified_tools:
                            print(f"Calling progress callback for tool: {tool_name}")
                            progress_callback(tool_messages[tool_name])
                            notified_tools.add(tool_name)
                else:
                    print("Progress callback is None, skipping notification")

            # Stream partial text content if available and streaming callback exists
            # Only stream AI messages (not system, user, or tool messages)
            elif hasattr(latest, 'content') and isinstance(latest.content, str) and streaming_callback:
                # Check if this is an AI message by checking the type
                from langchain_core.messages import AIMessage
                if isinstance(latest, AIMessage):
                    new_text = latest.content
                    if new_text and new_text != accumulated_text:
                        accumulated_text = new_text
                        print(f"Streaming partial content: {len(accumulated_text)} chars")
                        streaming_callback(accumulated_text)

    # Extract text response
    resp = final_state["messages"][-1].content if final_state and "messages" in final_state else ""

    # Extract image paths from tool messages
    image_paths = state.get("image_paths", []).copy()
    if final_state and "messages" in final_state:
        for msg in final_state["messages"]:
            # Check if message is a ToolMessage from generate_image
            if isinstance(msg, ToolMessage) and hasattr(msg, 'name') and msg.name == 'generate_image':
                image_paths.append(msg.content)

    return {'messages': [resp], 'image_paths': image_paths}


def get_config_values(config: RunnableConfig) -> RunnableConfig:
    metadata = config.get("metadata", {})
    config_values: RunnableConfig = {
        "configurable": {
            "user_id": metadata.get("user_id"),
            "thread_id": metadata.get("thread_id"),
        },
        "metadata": metadata  # Preserve full metadata including progress_callback
    }
    return config_values


workflow = StateGraph(EnhancedState)

workflow.add_node(CONVERSATION_NODE, conversation)
workflow.add_node(SUMMARIZE_CONVERSATION_NODE, summarize_conversation)

workflow.add_edge(START, CONVERSATION_NODE)
workflow.add_conditional_edges(CONVERSATION_NODE, should_continue)
workflow.add_edge(SUMMARIZE_CONVERSATION_NODE, END)

app = workflow.compile(checkpointer=checkpointer, store=store)

print(get_conversation_tools_description())

with open("mister_fritz_diagram.png", "wb") as binary_file:
    binary_file.write(app.get_graph().draw_mermaid_png())