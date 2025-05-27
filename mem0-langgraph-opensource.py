import asyncio
import os
import time
from dotenv import load_dotenv
from utils import get_llms
from mem0 import Memory

# Import LangGraph components
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

#untuk connect ke supabase
import psycopg2

load_dotenv() 

# Define your tools here. For demonstration, let's create a simple tool.
def get_current_time() -> str:
    """Returns the current time."""
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")

# Global variables
llm = None
mem0 = None

class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    mem0_user_id: str
    thinking_time: float

def chatbot(state: State):
    messages = state["messages"]
    user_id = state["mem0_user_id"]
    
    # Retrieve relevant memories
    memories = mem0.search(messages[-1].content, user_id=user_id)
    context = "Relevant information from previous conversations:\n"

    # Handle the correct structure: memories is a dict with 'results' key
    if memories and 'results' in memories and memories['results']:
        for memory_item in memories['results']:
            if isinstance(memory_item, dict):
                if 'memory' in memory_item:
                    context += f"- {memory_item['memory']}\n"
                elif 'text' in memory_item:
                    context += f"- {memory_item['text']}\n"
                elif 'content' in memory_item:
                    context += f"- {memory_item['content']}\n"
                else:
                    context += f"- {memory_item}\n"
            else:
                context += f"- {memory_item}\n"
    else:
        context += "No relevant memories found.\n"

    system_message = SystemMessage(content=f"""You are a helpful customer support assistant. Use the provided context to personalize your responses and remember user preferences and past interactions.
{context}""")

    # Ensure the system message is at the beginning of the messages sent to LLM
    full_messages = [system_message] + messages
    
    # Measure only the LLM thinking time
    start_time = time.time()
    response = llm.invoke(full_messages)
    end_time = time.time()
    thinking_time = end_time - start_time
    
    # Store the thinking time in the state to pass it back
    state["thinking_time"] = thinking_time

    # Store the interaction in Mem0
    # Store both user query and AI response for better context in future searches
    mem0.add(f"User: {messages[-1].content}\nAssistant: {response.content}", user_id=user_id)
    
    return {"messages": [response], "thinking_time": thinking_time}

def run_conversation(compiled_graph, user_input: str, mem0_user_id: str):
    """Run a single conversation turn"""
    config = {"configurable": {"thread_id": mem0_user_id}}
    state = {"messages": [HumanMessage(content=user_input)], "mem0_user_id": mem0_user_id}

    # Get the final result from the graph
    result = compiled_graph.invoke(state, config)
    
    if result and "messages" in result and len(result["messages"]) > 0:
        # Get the last message which should be the AI response
        ai_response = result["messages"][-1].content
        thinking_time = result.get("thinking_time", 0)
        return ai_response, thinking_time
    else:
        return "No response generated.", 0

async def main():
    global llm, mem0 # Declare global to assign
    print("Initializing LLM...")
    llm = get_llms() # Initialize LLM here
    
    mem0_config = {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "host": os.getenv('host'),
                "port": int(os.getenv('port')),
                "user": os.getenv('user'),     
                "password": os.getenv('password'),
                "dbname": os.getenv('dbname'),
                "embedding_model_dims" : 1536   
            }
        },
        "embedder": {
            "provider": "azure_openai",
            "config": {
                "model": os.getenv('AZURE_OPENAI_MODEL'),
                "embedding_dims": 1536,
                "azure_kwargs": {
                    "azure_deployment": os.getenv('AZURE_DEPLOYMENT'),
                    "api_version": os.getenv('AZURE_OPENAI_API_VERSION'),
                    "azure_endpoint": os.getenv('OPENAI_AZURE_EMBEDDINGS_ENDPOINT'),
                    "api_key": os.getenv('AZURE_OPENAI_API_KEY'),

                }
            }
        },
        "llm": {
            "provider": "openai",  # Use openai provider but with openrouter settings
            "config": {
                "api_key": "123123",
                "openai_base_url": os.getenv('OPEN_ROUTER_BASE_URL'),  # This should point to OpenRouter
                "model": "anthropic/claude-3.5-sonnet"
            }
        }
    }
    mem0 = Memory.from_config(mem0_config) # Initialize Mem0

    mem0.reset()  # Add this line to reset and recreate tables with correct dimensions
    print("Mem0 reset complete - tables recreated with 384 dimensions")

    # Define the LangGraph
    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END) # Simple flow: chatbot -> end

    compiled_graph = graph.compile()
    print("LangGraph compiled.")
    
    total_time_taken = 0 # Initialize total time

    # --- Conversation Simulation ---
    print("Welcome to Customer Support! How can I assist you today?")
    
    # Using fixed user IDs for demonstration, you'd generate/retrieve these
    user_id_1 = "customer_123" 
    user_id_2 = "customer_456"

    conversations = [
        # Conversation 1 (user_id_1)
        {"user_id": user_id_1, "query": "I'm Alergic to peanut and my name is max."},
        {"user_id": user_id_1, "query": "Why can i alergic to that?"},
        
        # Conversation 2 (user_id_2)
        {"user_id": user_id_2, "query": "What is the capital of France?"},
        {"user_id": user_id_2, "query": "And its population?"},

        # Back to Conversation 1 (user_id_1) to show memory recall
        {"user_id": user_id_1, "query": "What's my name and am i alergic to something?."},
    ]

    for i, convo in enumerate(conversations):
        user_input = convo["query"]
        current_user_id = convo["user_id"]
        
        print(f"\n--- Query {i+1} (User: {current_user_id}) ---")
        print(f"You: {user_input}")
        
        # Run the conversation and get both response and thinking time
        ai_response, thinking_time = run_conversation(compiled_graph, user_input, current_user_id)
        
        total_time_taken += thinking_time

        print(f"Customer Support: {ai_response}")
        print(f"Thinking time: {thinking_time:.2f} seconds")

    print(f"\n--- Total thinking time for all queries: {total_time_taken:.2f} seconds ---")

if __name__ == "__main__":
    asyncio.run(main())
