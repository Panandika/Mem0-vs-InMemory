import asyncio
import os
import time # Import time module
from dotenv import load_dotenv
from utils import get_llms

# Import LangGraph components for a custom graph
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv() 

# Global variables
llm = None

class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    thinking_time: float

def chatbot(state: State):
    """Process messages and generate response"""
    messages = state["messages"]
    
    # Create system message for the assistant
    system_message = SystemMessage(content="""You are a helpful customer support assistant. Use the conversation history to personalize your responses and remember user preferences and past interactions.""")

    # Combine system message with conversation history
    full_messages = [system_message] + messages
    
    # Measure only the LLM thinking time
    start_time = time.time()
    response = llm.invoke(full_messages)
    end_time = time.time()
    thinking_time = end_time - start_time
    
    return {"messages": [response], "thinking_time": thinking_time}

def run_conversation(compiled_graph, user_input: str, user_id: str):
    """Run a single conversation turn"""
    config = {"configurable": {"thread_id": user_id}}
    state = {"messages": [HumanMessage(content=user_input)]}

    result = compiled_graph.invoke(state, config)
    
    if result and "messages" in result and len(result["messages"]) > 0:
        # Get the last message which should be the AI response
        ai_response = result["messages"][-1].content
        thinking_time = result.get("thinking_time", 0)
        return ai_response, thinking_time
    else:
        return "No response generated.", 0

async def main():
    global llm
    
    print("Initializing LLM...")
    llm = get_llms()
    
    # Initialize InMemorySaver for checkpointing
    checkpointer = InMemorySaver()
    print("In-memory checkpointer initialized.")

    # Define the LangGraph with checkpointer for memory
    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    # Compile with checkpointer to enable memory
    compiled_graph = graph.compile(checkpointer=checkpointer)
    print("LangGraph compiled with memory support.")
    
    total_time_taken = 0

    print("Welcome to Customer Support! How can I assist you today?")
    
    user_id_1 = "customer_123" 
    user_id_2 = "customer_456"

    conversations = [
        {"user_id": user_id_1, "query": "I'm Alergic to peanut and my name is max."},
        {"user_id": user_id_1, "query": "Why can i alergic to that?"},
        {"user_id": user_id_2, "query": "What is the capital of France?"},
        {"user_id": user_id_2, "query": "And its population?"},
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
    
    # Optional: Print checkpointer contents for debugging
    print(f"\n--- Memory Store Contents ---")
    try:
        # Get all stored thread IDs from the checkpointer
        stored_threads = list(checkpointer.storage.keys())
        print(f"Stored conversation threads: {len(stored_threads)}")
        for thread_id in [user_id_1, user_id_2]:
            thread_key = f"thread:{thread_id}"
            if any(thread_id in str(key) for key in stored_threads):
                print(f"User {thread_id}: conversation history stored")
    except Exception as e:
        print(f"Could not access checkpointer contents: {e}")

if __name__ == "__main__":
    asyncio.run(main())
