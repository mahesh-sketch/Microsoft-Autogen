
# Swarm implements a team in which agents can hand off task to other agents based on their capabilities. 
# It is a multi-agent design pattern first introduced by OpenAI in Swarm. The key idea is 
# to let agent delegate tasks to other agents using a special tool call, while all agents share the same message context. 

import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console

from dotenv import load_dotenv
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
    model="gpt-4o-2024-11-20",
    azure_deployment=openai_deployment,
    api_key=open_api_key,
    api_version=open_api_version,
    azure_endpoint=openai_endpoint,
)

def refund_flight(flight_id:str) -> str:
    return f"Flight {flight_id} refunded"

travel_agent = AssistantAgent(
    "travel_agent",
    model_client=model_client,
    handoffs=["flights_refunder", "user"],
    system_message="""You are a travel agent.
    The flights_refunder is in charge of refunding flights.
    If you need information from the user, you must first send your message, then you can handoff to the user.
    Use TERMINATE when the travel planning is complete.""",
)

flights_refunder = AssistantAgent(
    "flights_refunder",
    model_client=model_client,
    handoffs=["travel_agent", "user"],
    tools=[refund_flight],
    system_message="""You are an agent specialized in refunding flights.
    You only need flight reference numbers to refund a flight.
    You have the ability to refund a flight using the refund_flight tool.
    If you need information from the user, you must first send your message, then you can handoff to the user.
    When the transaction is complete, handoff to the travel agent to finalize.""",
)

termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE")
team = Swarm([travel_agent,flights_refunder],termination_condition=termination)

task = "I need to refund my flight."


async def run_team_stream() -> None:
    task_result = await Console(team.run_stream(task=task))
    last_message = task_result.messages[-1]

    while isinstance(last_message, HandoffMessage) and last_message.target == "user":
        user_message = input("User: ")

        task_result = await Console(
            team.run_stream(task=HandoffMessage(source="user", target=last_message.source, content=user_message))
        )
        last_message = task_result.messages[-1]

async def main():
    await run_team_stream()

if __name__ == "__main__":
    asyncio.run(main())
