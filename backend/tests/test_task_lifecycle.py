import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from app.utils.task_lifecycle import BackgroundTaskManager
from app.main import lifespan, app, periodic_news_fetcher

pytestmark = pytest.mark.unit

async def test_task_manager_start_and_track():
    """1. Test that BackgroundTaskManager registers and tracks active tasks."""
    tm = BackgroundTaskManager()

    async def sample_task():
        await asyncio.sleep(5)

    task = tm.create_task(sample_task(), name="test_sample")
    assert task in tm.active_tasks
    assert len(tm.active_tasks) == 1

    # Clean up
    await tm.cancel_all()
    assert len(tm.active_tasks) == 0


async def test_task_manager_cancellation_and_no_orphans():
    """2. Test that BackgroundTaskManager cancels active tasks and prevents orphaned tasks."""
    tm = BackgroundTaskManager()
    task_ran = False
    task_cancelled = False

    async def long_running_worker():
        nonlocal task_ran, task_cancelled
        task_ran = True
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            task_cancelled = True
            raise

    task = tm.create_task(long_running_worker(), name="test_worker")
    await asyncio.sleep(0.01)
    assert task_ran is True
    assert task in tm.active_tasks

    await tm.cancel_all()

    assert task_cancelled is True
    assert task.done() is True
    assert len(tm.active_tasks) == 0, "No orphaned tasks should remain"


async def test_periodic_news_fetcher_cancellation_handling():
    """3. Test that periodic_news_fetcher handles asyncio.CancelledError cleanly."""
    with patch("app.main.mcp_client.call_tool", new_callable=AsyncMock):
        task = asyncio.create_task(periodic_news_fetcher())
        await asyncio.sleep(0.01)
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.done() is True


async def test_fastapi_lifespan_startup_and_shutdown():
    """4. Test FastAPI lifespan context manager startup registration and clean task shutdown."""
    with patch("app.main.mcp_client.start", new_callable=AsyncMock), \
         patch("app.main.mcp_client.list_available_tools", new_callable=AsyncMock, return_value=["search_live_news"]), \
         patch("app.main.mcp_client.call_tool", new_callable=AsyncMock), \
         patch("app.main.mcp_client.stop", new_callable=AsyncMock):

        from app.utils.task_lifecycle import task_manager

        async with lifespan(app):
            # Startup checks
            assert len(task_manager.active_tasks) >= 1, "Background tasks must be registered on startup"

        # Shutdown checks
        assert len(task_manager.active_tasks) == 0, "All background tasks must be cleanly shut down on app exit"
