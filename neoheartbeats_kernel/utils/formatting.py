import json
from collections.abc import Callable
from typing import Any, Literal, cast
from pydantic import BaseModel, Field


# OpenAI API compatible models
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


def format_messages(messages: list[dict[str, str]]) -> list[Message]:
    """Format a list of message dictionaries into Message objects."""
    return [
        Message(
            role=cast(Literal["system", "user", "assistant"], msg["role"]),
            content=msg["content"],
        )
        for msg in messages
    ]


def format_chat_completion(
    id: str, created: int, model: str, messages: list[Message], total_tokens: int
) -> ChatCompletion:
    """Format a chat completion response."""
    return ChatCompletion(
        id=id,
        created=created,
        model=model,
        choices=[{"index": 0, "message": messages[-1], "finish_reason": "stop"}],
        usage={
            "prompt_tokens": sum(len(msg.content.split()) for msg in messages[:-1]),
            "completion_tokens": len(messages[-1].content.split()),
            "total_tokens": total_tokens,
        },
    )


def parse_openai_request(request_data: dict[str, Any]) -> tuple[str, list[Message]]:
    """Parse an OpenAI-compatible request."""
    model = request_data.get("model", "")
    messages = format_messages(request_data.get("messages", []))
    return model, messages


def create_openai_response(
    id: str, created: int, model: str, messages: list[Message], total_tokens: int
) -> dict[str, Any]:
    """Create an OpenAI-compatible response."""
    completion = format_chat_completion(id, created, model, messages, total_tokens)
    return completion.model_dump()


def apply_function_to_messages(
    messages: list[Message], func: Callable[[str], str]
) -> list[Message]:
    """Apply a function to the content of each message."""
    return [Message(role=msg.role, content=func(msg.content)) for msg in messages]


def filter_messages_by_role(
    messages: list[Message], roles: set[Literal["system", "user", "assistant"]]
) -> list[Message]:
    """Filter messages by role."""
    return [msg for msg in messages if msg.role in roles]


def combine_message_contents(messages: list[Message], separator: str = " ") -> str:
    """Combine the contents of multiple messages."""
    return separator.join(msg.content for msg in messages)
