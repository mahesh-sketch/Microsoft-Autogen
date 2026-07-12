import asyncio
from dotenv import load_dotenv
import os
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from langchain_community.utilities import GoogleSerperAPIWrapper

from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential


load_dotenv()

vault_url = os.getenv("AZURE_VAULT_URL")

credential = DefaultAzureCredential()

client = SecretClient(credential=credential, vault_url=vault_url)

open_api_key = client.get_secret("openai-api-key").value
open_api_version = client.get_secret("openai-api-version").value
openai_deployment = client.get_secret("openai-deployment").value
openai_endpoint = client.get_secret("openai-endpoint").value

# Get SerperAPI key from Key Vault and set as environment variable
try:
    serper_api_key = os.getenv("SERPER_API_KEY")
    print("[INFO] SerperAPI key loaded from Key Vault")
except Exception as e:
    print(f"[WARNING] Failed to load SerperAPI key from Key Vault: {e}")


model_client = AzureOpenAIChatCompletionClient(
    model="gpt-4o-2024-11-20",
    azure_deployment=openai_deployment,
    api_key=open_api_key,
    api_version=open_api_version,
    azure_endpoint=openai_endpoint,
)

search_tool_wrapper = GoogleSerperAPIWrapper(type='news')

# Create a simple function to use as a tool
def search_web(query: str) -> str:
    """Search the web using Serper API"""
    try:
        print(f"[DEBUG] Executing search for: {query}")
        result = search_tool_wrapper.run(query)
        print(f"[DEBUG] Search result length: {len(result)} characters")
        print(f"[DEBUG] Result preview: {result[:200]}...")
        return result
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        print(f"[ERROR] SerperAPI error: {error_msg}")
        return error_msg

search_agent = AssistantAgent(
    name='SearchAgent',
    model_client=model_client,
    system_message="""You are a helpful assistant that can search the web to find current information. 
    When asked a question, use the search_web tool to find relevant information and provide a comprehensive answer based on the search results.""",
    description='Searches the internet and provides detailed answers based on search results',
    tools=[search_web],
    reflect_on_tool_use=True
)

async def demonstrate_search():
    """Demonstrate the search functionality"""
    print("=== AutoGen Third-Party Tools Demonstration ===\n")
    
    # Test queries for demonstration
    test_queries = [
        "Who won the last IPL tournament in cricket in 2025?",
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 50)
        
        try:
            # Run the agent with the query
            result = await search_agent.run(task=query)
            print(f"Response: {result.messages[-1].content}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "="*70 + "\n")


async def main():
    await demonstrate_search()

if __name__ == "__main__":
    asyncio.run(main())