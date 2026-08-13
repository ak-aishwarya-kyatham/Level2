import asyncio
import logging
from typing import Set, Optional

logger = logging.getLogger(__name__)

class BackgroundTaskManager:
    """
    Manages the lifecycle of asyncio background tasks created during FastAPI app startup.
    Ensures tasks are tracked, safely cancelled on shutdown, and cleanly awaited to prevent orphaned tasks.
    """
    def __init__(self):
        self._tasks: Set[asyncio.Task] = set()

    def create_task(self, coro, name: Optional[str] = None) -> asyncio.Task:
        """Schedules a coroutine as an asyncio.Task and registers it for lifecycle tracking."""
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    @property
    def active_tasks(self) -> Set[asyncio.Task]:
        """Returns currently active managed tasks."""
        return {t for t in self._tasks if not t.done()}

    async def cancel_all(self, timeout: float = 5.0):
        """Cancels all registered background tasks and awaits completion safely."""
        active = self.active_tasks
        if not active:
            return

        logger.info(f"[Task Lifecycle Manager] Cancelling {len(active)} active background task(s)...")
        for task in active:
            task.cancel()

        results = await asyncio.gather(*active, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                logger.error(f"[Task Lifecycle Manager Error]: Uncaught exception in cancelled task: {res}")

        self._tasks.clear()
        logger.info("[Task Lifecycle Manager] All background tasks cleanly shut down.")

# Singleton instance for application lifecycle management
task_manager = BackgroundTaskManager()
