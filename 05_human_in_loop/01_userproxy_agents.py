import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
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
    name='Assistant',
    description='you are a great assistant',
    model_client=model_client,
    system_message='You are a really helpful assistant who help on the task given.'
)


user_agent = UserProxyAgent(
    name="UserProxy",
    description='A proxy agent that represent a user',
    input_func=input
)


termination_condition = TextMentionTermination('APPROVE')

team = RoundRobinGroupChat(
    participants=[assistant,user_agent],
    termination_condition=termination_condition
)

stream = team.run_stream(task='Write a nice 4 line poem about India')

async def main():
    await Console(stream)


if(__name__=="__main__"):
    asyncio.run(main())