import asyncio
from abc import ABC, abstractmethod
import json
from typing import AsyncGenerator, Any, Dict, List

# Base loader with shared interface
class BaseDatasetLoader(ABC):
    def __init__(self, config: Dict[str, Any], processor: Any):
        self.config = config
        self.queue = asyncio.Queue(maxsize=config.get("queue_size", 64))
        self.processor = processor

    @abstractmethod
    async def load_data(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Load a batch of documents from the dataset.
           Return None when there are no more batches."""
        pass

    @abstractmethod
    def eval(self, pred: str, ans: str) -> float:
        """Return 1.0 if pred matches ans, 0.0 otherwise."""
        return float(pred == ans)

    async def producer(self):
        """Continuously load batches and put each document into the queue."""
        async for item in self.load_data():
            await self.queue.put(item)
            print(self.queue.qsize())
            
        # Signal termination for all consumers
        for _ in range(self.config.get("num_workers", 4)):
            await self.queue.put(None)

    async def consumer(self):
        """Consume items from the queue and process them."""
        task_name = asyncio.current_task().get_name()
        while True:
            item = await self.queue.get()
            if item is None:
                print("Stop!")
                break
            await self.processor(**item)

    async def run(self):
        """Run the producer-consumer pipeline."""
        producer_task = asyncio.create_task(self.producer(), name="Producer")
        consumer_tasks = [
            asyncio.create_task(self.consumer(), name=f"Consumer-{i}")
            for i in range(self.config.get("num_workers", 4))
        ]
        await asyncio.gather(producer_task, *consumer_tasks)