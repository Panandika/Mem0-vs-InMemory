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
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

#untuk connect ke supabase
import psycopg2

load_dotenv() 

# Define your tools here. For demonstration, let's create a simple tool.
@tool
def get_current_time() -> str:
    """Returns the current time."""
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")

# Global variables
llm = None
mem0 = None
agent = None

class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    thinking_time: float

async def run_conversation(agent, user_input: str, mem0_user_id: str):
    """Run a single conversation turn"""
    config = {"configurable": {"thread_id": mem0_user_id}}
    
    input_messages = [HumanMessage(content=user_input)]

    # Measure only the LLM thinking time (including tool use)
    start_time = time.time()
    
    result = await agent.ainvoke(
        {"messages": input_messages},
        config
    )
    end_time = time.time()
    thinking_time = end_time - start_time
    
    if result and "messages" in result and len(result["messages"]) > 0:
        ai_response = result["messages"][-1].content
        
        # Store the conversation in Mem0
        conversation_messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ai_response}
        ]
        mem0.add(conversation_messages, user_id=mem0_user_id)
        
        return ai_response, thinking_time
    else:
        return "No response generated.", 0

async def main():
    global llm, mem0, agent 
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
                "embedding_model_dims" : 3072   
            }
        },
        "embedder": {
            "provider": "azure_openai",
            "config": {
                "model": os.getenv('AZURE_OPENAI_MODEL'),
                "embedding_dims": 3072,
                "azure_kwargs": {
                    "azure_deployment": os.getenv('AZURE_DEPLOYMENT'),
                    "api_version": os.getenv('AZURE_OPENAI_API_VERSION'),
                    "azure_endpoint": os.getenv('OPENAI_AZURE_EMBEDDINGS_ENDPOINT'),
                    "api_key": os.getenv('AZURE_OPENAI_API_KEY'),

                }
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "api_key": os.getenv('OPEN_ROUTER_API_KEY'),  
                "openai_base_url": os.getenv('OPEN_ROUTER_BASE_URL'),
                "model": os.getenv('OPEN_ROUTER_MEM0_MODEL')
            }
        }        
    }
    mem0 = Memory.from_config(mem0_config) # Initialize Mem0

    #mem0.reset()  # Add this line to reset and recreate tables with correct dimensions
    #print("Mem0 reset complete")

    # Define the tools the agent can use
    tools = [get_current_time]

    # Create the agent using create_react_agent
    # We need a custom prompt to integrate Mem0's context
    def get_agent_prompt(state, config=None):
        messages = state["messages"]
        
        # Get user_id from config instead of state
        user_id = None
        if config and "configurable" in config:
            user_id = config["configurable"].get("thread_id")  # We're using thread_id as user_id
        
        if not user_id:
            # Fallback if no user_id is available
            context = "No previous conversation history available.\n"
        else:
            # Retrieve relevant memories
            memories = mem0.search(messages[-1].content, user_id=user_id)
            context = "Relevant information from previous conversations:\n"

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

        system_message_content = f"""You are a helpful customer support assistant. Use the provided context to personalize your responses and remember user preferences and past interactions.
{context}

You have access to the following tools:
{tools}

If the user asks for the current time, use the 'get_current_time' tool.
"""
        return [SystemMessage(content=system_message_content)] + messages

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=get_agent_prompt, # Use the dynamic prompt
        # We need a checkpointer for memory, but Mem0 handles the long-term memory.
        # For short-term conversation memory within LangGraph, InMemorySaver can be used.
        # However, since Mem0 is handling the primary memory, we might not strictly need
        # a LangGraph checkpointer for this specific setup if Mem0 is the source of truth.
        # Let's omit it for now to keep it simple and rely on Mem0.
    )

   
    total_time_taken = 0 # Initialize total time

    # --- Conversation Simulation ---
    print("Welcome to Customer Support! How can I assist you today?")
    
    # Using fixed user IDs for demonstration, you'd generate/retrieve these
    user_id_1 = "customer_123" 
    user_id_2 = "customer_456"

    conversations = [
        # --- User 1: Initial Information & Recall ---
        {"user_id": user_id_1, "query": "My name is Alice and I love gardening. My favorite flower is a rose."},
        {"user_id": user_id_1, "query": "What's my name and what do I like to do?"},
        {"user_id": user_id_1, "query": "What is my favorite flower?"},

        # --- User 1: Evolving Information & Related Entity ---
        # This tests if Mem0 can associate new info with Alice and retrieve it.
        {"user_id": user_id_1, "query": "I recently adopted a cat named Whiskers. Whiskers loves to play with yarn."},
        {"user_id": user_id_1, "query": "Tell me about my pet."},
        {"user_id": user_id_1, "query": "What does Whiskers like to do?"},

        # --- User 2: Separate Context ---
        {"user_id": user_id_2, "query": "I'm Bob and I'm a software engineer. I prefer coffee over tea."},
        {"user_id": user_id_2, "query": "What's my profession and beverage preference?"},

        # --- Context Switching & Recall ---
        # This is crucial to test user_id separation.
        {"user_id": user_id_1, "query": "Remind me about my favorite flower and my pet's name."},
        {"user_id": user_id_2, "query": "What drink do I prefer?"},
        {"user_id": user_id_1, "query": "What is the current time?"}, # Keep a tool test
    ]

    for i, convo in enumerate(conversations):
        user_input = convo["query"]
        current_user_id = convo["user_id"]
        
        print(f"\n--- Query {i+1} (User: {current_user_id}) ---")
        print(f"You: {user_input}")
        
        # Run the conversation and get both response and thinking time
        ai_response, thinking_time = await run_conversation(agent, user_input, current_user_id)
        
        total_time_taken += thinking_time

        print(f"Customer Support: {ai_response}")
        print(f"Thinking time: {thinking_time:.2f} seconds")

    print(f"\n--- Total thinking time for all queries: {total_time_taken:.2f} seconds ---")

if __name__ == "__main__":
    asyncio.run(main())
