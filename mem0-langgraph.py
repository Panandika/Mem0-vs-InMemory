import asyncio
import os
from mem0 import MemoryClient
import time # Import time module
from dotenv import load_dotenv
from utils import get_llms

# Import LangGraph components for a custom graph
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI # Assuming get_llms returns a ChatOpenAI instance or compatible

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

    if memories:
        for memory_item in memories:
            # Mem0 search returns 'memory' key for the content
            context += f"- {memory_item['memory']}\n"
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

    MEM0_KEY = os.getenv('MEM0_KEY')
    mem0 = MemoryClient(api_key=MEM0_KEY) # Initialize Mem0

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
        {"user_id": user_id_1, "query": "What's my name and am i alergic to something?"},
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
