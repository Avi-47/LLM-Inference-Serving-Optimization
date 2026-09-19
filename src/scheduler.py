import asyncio
import uuid


class Request:
    def __init__(self, prompt, max_tokens):
        self.id = str(uuid.uuid4())
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.generator = None
        self.queue = asyncio.Queue()


class RequestManager:
    def __init__(self, generator_fn, max_batch_size=8):
        self.waiting = []
        self.running = {}
        self.max_batch_size = max_batch_size
        self.generator_fn = generator_fn

    def submit(self, request):
        self.waiting.append(request)

    def admit_waiting(self):
        while self.waiting and len(self.running) < self.max_batch_size:
            req = self.waiting.pop(0)
            req.generator = self.generator_fn(req, self.waiting)
            self.running[req.id] = req

    async def step(self):
        finished_ids = []
        for req_id, req in list(self.running.items()):
            try:
                token_id = next(req.generator)
                text = tokenizer.decode([token_id])
                await req.queue.put(text)
                await asyncio.sleep(0)
            except StopIteration:
                finished_ids.append(req_id)
        for req_id in finished_ids:
            req = self.running.pop(req_id)
            await req.queue.put(None)


async def scheduler_loop(manager):
    while True:
        manager.admit_waiting()
        if manager.running:
            await manager.step()
        else:
            await asyncio.sleep(0.01)
