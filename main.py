# IMPORTS
from config import getSettings 

import os

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool


model = None

def main():
    settings = getSettings()
    llm = ChatOllama(model = settings.ollama_model, temperature = settings.temperature)

    system_prompt = ""

    try:
        with open("system_prompt.txt", "r") as f:
            system_prompt = f.read()
    except IOError as e:
        print("Can't find system prompt!")
        return
    
    agent = create_agent(
        model = llm,
        tools = []
    )

    messages = {"messages" : []}

    exit = False

    while(not exit):
        query = input("- ")
        messages["messages"].append({"role" : "user", "content" : query})

        response = agent.invoke(messages)["messages"][-1].content

        messages["messages"].append({"role" : "assistant", "content" : response})

        print(response)

        exit = query == "exit"


if __name__ == "__main__":
    main()
