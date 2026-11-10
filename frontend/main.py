# AI Generated Demo, do not read

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict, Union
import zmq
import uvicorn
import logging
import time
import uuid
import json

app = FastAPI(title="Forch Inference Frontend (OpenAI-compatible)")

# Setup ZMQ Context and Socket
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")  # Assume backend listens on this port

logger = logging.getLogger("uvicorn.error")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = "forch-small"
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 64
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: Optional[str] = "forch-small"
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 64
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1


def build_prompt_from_messages(messages: List[ChatMessage]) -> str:
    parts = []
    for m in messages:
        parts.append(f"{m.role}: {m.content}")
    return "\n".join(parts)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    prompt = build_prompt_from_messages(req.messages)
    logger.info(f"chat_completions received prompt length={len(prompt)} model={req.model}")

    zmq_payload = {
        "type": "chat.completion",
        "model": req.model,
        "prompt": prompt,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "n": req.n,
    }

    try:
        socket.send_json(zmq_payload)
        zmq_reply = socket.recv_string()
    except Exception as e:
        logger.exception("ZMQ request failed")
        raise HTTPException(status_code=502, detail=str(e))

    # Try to parse backend reply as JSON; fallback to raw string
    try:
        content = json.loads(zmq_reply)
        # If backend returns structured object, try to extract text
        if isinstance(content, dict) and "completion" in content:
            text = content.get("completion")
        else:
            text = zmq_reply
    except Exception:
        text = zmq_reply

    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
    }

    return response


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if isinstance(req.prompt, list):
        prompt = "\n".join(req.prompt)
    else:
        prompt = req.prompt

    logger.info(f"completions received prompt length={len(prompt)} model={req.model}")

    zmq_payload = {
        "type": "text.completion",
        "model": req.model,
        "prompt": prompt,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "n": req.n,
    }

    try:
        socket.send_json(zmq_payload)
        zmq_reply = socket.recv_string()
    except Exception as e:
        logger.exception("ZMQ request failed")
        raise HTTPException(status_code=502, detail=str(e))

    try:
        content = json.loads(zmq_reply)
        if isinstance(content, dict) and "completion" in content:
            text = content.get("completion")
        else:
            text = zmq_reply
    except Exception:
        text = zmq_reply

    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    response = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": req.model,
        "choices": [
            {"text": text, "index": 0, "finish_reason": "stop"}
        ],
        "usage": None,
    }

    return response


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
