from typing import Any
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

# Registry for LLM tools
tool_registry = {}


def llm_tool(name: str):
    def decorator(func):
        tool_registry[name] = func
        return func

    return decorator


@dataclass
class LLMToolFunctionParams:
    type: str
    properties: dict[str, Any]
    required: list[str]
    additionalProperties: bool


@dataclass
class LLMToolFunction:
    name: str
    description: str
    parameters: LLMToolFunctionParams


@dataclass
class LLMTool:
    type: str
    function: LLMToolFunction


@dataclass
class SamplingParams:
    model: str = "gpt-4o-mini"
    temperature: float = 0.65
    top_p: float = 0.90


@dataclass
class LLMChat:
    base_url: str = "https://api.openai.com/v1/chat/completions"
    api_key: str | None = None

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }


def llm_message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def post_request(
    data: dict[str, Any], headers: dict[str, str], url: str
) -> dict[str, Any]:
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()


@llm_tool("get_current_time")
def get_current_time() -> dict[str, Any]:
    return {"result": datetime.now().isoformat()}


def call_tool(tool_call: dict[str, Any], tools: list[LLMTool]) -> dict[str, Any]:
    tool_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])
    if tool_name in tool_registry:
        return tool_registry[tool_name](**arguments)
    tool_dict = {tool.function.name: tool for tool in tools}
    if tool_name in tool_dict:
        return {"result": f"Called tool {tool_name} with parameters {arguments}"}
    raise ValueError(f"Tool {tool_name} not found")


def chat_synthesize(
    chat_object: LLMChat,
    messages: list[dict[str, str]],
    sampling_params: SamplingParams,
    tools: list[LLMTool] | None = None,
) -> str:
    data = {
        "model": sampling_params.model,
        "messages": messages,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
    }

    if tools:
        data.update(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.function.name,
                            "description": tool.function.description,
                            "parameters": asdict(tool.function.parameters),
                        },
                    }
                    for tool in tools
                ],
                "tool_choice": "auto",
            }
        )
        completion = post_request(data, chat_object.headers, chat_object.base_url)
        tool_call = completion["choices"][0]["message"]["tool_calls"][0]
        tool_get = call_tool(tool_call, tools)
        function_call_result_message = {
            "role": "tool",
            "content": json.dumps({"result": tool_get["result"]}),
            "tool_call_id": tool_call["id"],
        }

        messages.extend(
            [completion["choices"][0]["message"], function_call_result_message]
        )
        return chat_synthesize(
            chat_object=chat_object,
            messages=messages,
            sampling_params=sampling_params,
            tools=None,
        )

    completion = post_request(data, chat_object.headers, chat_object.base_url)
    return completion["choices"][0]["message"]["content"]


# Test client
if __name__ == "__main__":
    # Initialize ChatObject with a dummy API key
    chat_object = LLMChat(api_key=os.getenv("OPENAI_API_KEY"))

    # Define a tool to get the current time
    tool_function_params = LLMToolFunctionParams(
        type="object",
        properties={},
        required=[],
        additionalProperties=False,
    )
    tool_function = LLMToolFunction(
        name="get_current_time",
        description="Returns the current time",
        parameters=tool_function_params,
    )
    tool = LLMTool(type="tool", function=tool_function)
    sampling_params = SamplingParams()

    # Call chat_synthesize with user input
    user_input = "What is the current time?"
    response = chat_synthesize(
        chat_object=chat_object,
        messages=[llm_message("user", user_input)],
        sampling_params=sampling_params,
        tools=[tool],
    )

    # Print the response
    print(response)
