import asyncio
from autogen_agentchat.agents import AssistantAgent

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from dotenv import load_dotenv
from autogen_agentchat.ui import Console
import os

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
    model="gpt-4o",
    azure_deployment=openai_deployment,
    api_key=open_api_key,
    api_version=open_api_version,
    azure_endpoint=openai_endpoint,
)


assistant = AssistantAgent(
    name='Writer',
    description='you are a great writer',
    model_client=model_client,
    system_message='You are a really helpful writer who writes in less then 30 words.'
)

assistant2 = AssistantAgent(
    name='Reviewer',
    description='you are a great reviewer',
    model_client=model_client,
    system_message='You are a really helpful reviewer who writes in less then 30 words..'
)

assistant3 = AssistantAgent(
    name='Editor',
    description='you are a great editor',
    model_client=model_client,
    system_message='You are a really helpful editor who writes in less then 30 words..'
)


team = RoundRobinGroupChat(
    participants=[assistant,assistant2,assistant3],
    max_turns = 3
)


async def main():
    task = ' Write a 3 line poem about sky'

    while True:
        stream = team.run_stream(task=task)
        await Console(stream)

        feedback = input('Please Provide your feedback (type "exit" to stop)')
        if(feedback.lower().strip()=='exit'):
            break

        task = feedback

if (__name__=="__main__"):
    asyncio.run(main())