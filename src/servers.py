import asyncio
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from decode import generate_tokens
from scheduler import Request, RequestManager, scheduler_loop
from combined_pipeline import combined_generate


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 20


# server 1: no scheduling, no batching, no caching, no speculation

app = FastAPI()

@app.post("/generate")
def generate_endpoint(request: GenerateRequest):
    def token_stream():
        for token_id in generate_tokens(target_model, tokenizer, request.prompt, request.max_tokens):
            text = tokenizer.decode([token_id])
            yield f"data: {text}\n\n"
    return StreamingResponse(token_stream(), media_type="text/event-stream")

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


# server 2: round-robin scheduling only

def plain_generator_fn(req, waiting):
    return generate_tokens(target_model, tokenizer, req.prompt, req.max_tokens)

manager_v1 = RequestManager(plain_generator_fn, max_batch_size=8)
app_v1 = FastAPI()

@app_v1.on_event("startup")
async def startup_event_v1():
    asyncio.create_task(scheduler_loop(manager_v1))

@app_v1.post("/generate")
async def generate_endpoint_v1(request: GenerateRequest):
    req = Request(request.prompt, request.max_tokens)
    manager_v1.submit(req)

    async def event_stream():
        while True:
            text = await req.queue.get()
            if text is None:
                break
            yield f"data: {text}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

def run_server_v1():
    uvicorn.run(app_v1, host="0.0.0.0", port=8001, log_level="warning")


# server 3: round-robin scheduling + prefix caching + speculative decoding

CURRENT_K = 4

def combined_generator_fn(req, waiting):
    waiting_prompts = [r.prompt for r in waiting]
    return combined_generate(target_model, draft_model, tokenizer, req.prompt, waiting_prompts,
                              use_prefix_cache=True, use_speculative=True, k=CURRENT_K, max_tokens=req.max_tokens)

manager_combined = RequestManager(combined_generator_fn, max_batch_size=8)
app_combined = FastAPI()

@app_combined.on_event("startup")
async def startup_event_combined():
    asyncio.create_task(scheduler_loop(manager_combined))

@app_combined.post("/generate")
async def generate_endpoint_combined(request: GenerateRequest):
    req = Request(request.prompt, request.max_tokens)
    manager_combined.submit(req)

    async def event_stream():
        while True:
            text = await req.queue.get()
            if text is None:
                break
            yield f"data: {text}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

def run_server_combined():
    uvicorn.run(app_combined, host="0.0.0.0", port=8002, log_level="warning")


def start_all_servers():
    threading.Thread(target=run_server, daemon=True).start()
    threading.Thread(target=run_server_v1, daemon=True).start()
    threading.Thread(target=run_server_combined, daemon=True).start()
