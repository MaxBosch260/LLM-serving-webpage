import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")

# Holds the loaded model and tokenizer so that they are shared across requests instead of being reloaded on every call.
model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the container starts, not per request.
    model_state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_state["model"] = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32
    ).to("cpu")
    model_state["model"].eval()
    yield
    model_state.clear()


app = FastAPI(lifespan=lifespan)
INDEX_PATH = Path(__file__).with_name("index.html")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_PATH)


class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 100


@app.get("/health")
def health():
    # Useful later for an ALB target group or ECS container health check.
    return {"status": "ok"}


@app.post("/generate")
def generate(req: ChatRequest):
    if req.max_new_tokens <= 0 or req.max_new_tokens > 2048:
        raise ValueError("max_new_tokens must be between 1 and 2048")
    
    tokenizer = model_state["tokenizer"]
    model = model_state["model"]

    messages = [{"role": "user", "content": req.message}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=req.max_new_tokens)

    response_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )
    return {"response": response_text}