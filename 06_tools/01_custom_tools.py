import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
import os
from dotenv import load_dotenv

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


model_client = AzureOpenAIChatCompletionClient(
    model="gpt-4o-2024-11-20",
    azure_deployment=openai_deployment,
    api_key=open_api_key,
    api_version=open_api_version,
    azure_endpoint=openai_endpoint,
)


# Define a custom function to reverse a string
def reverse_string(text: str) -> str:
    """Reverse the given text.
        input : str


        output : str 
        The reversed String is returned.
    """
    return text[::-1]


# Register the custom function as a tool
reverse_tool = FunctionTool(reverse_string,description='A tool to reverse a string')

# Create an agent with the custom tool
agent = AssistantAgent(
    name="ReverseAgent",
    model_client=model_client,
    system_message="You are a helpful assistant that can reverse text using the reverse_string tool. Give the result with the summary",
    tools=[reverse_tool],
    reflect_on_tool_use=True,
)

# Define a task
task = "Reverse the text 'Hello, how are you Doing?'"

# Run the agent
async def main():
    result = await agent.run(task=task)

    # print(result)    
    print(f"Agent Response: {result.messages[-1].content}")

    print(reverse_string('Hello, how are you Doing?'))

if __name__ == "__main__":
    asyncio.run(main())