import os
from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
load_dotenv()

def get_llms(model_name: str="anthropic/claude-3.5-sonnet"): 
    """
    Helper function to get OpenAI model instance with necessary API key and endpoint.
    
    Args:
        model_name: The name of the model to use
        
    Returns:
        A configured ChatOpenAI instance
    """
    return ChatOpenAI(
        openai_api_key=os.getenv(
            "OPEN_ROUTER_API_KEY"
        ),
        openai_api_base=os.getenv(
            "OPEN_ROUTER_BASE_URL", 
            "https://openrouter.ai/api/v1/chat/completions"
        ),
        model=model_name,
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        streaming=True
    )
