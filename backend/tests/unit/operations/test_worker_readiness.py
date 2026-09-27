"""Readiness checks process-local queue tasks as well as persisted heartbeats."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


class WorkerReadinessContract(unittest.IsolatedAsyncioTestCase):
    async def test_ready_fails_when_a_queue_poller_has_stopped(self) -> None:
        """A fresh database heartbeat cannot conceal a dead durable-job poller."""
        from app import main

        saver = SimpleNamespace(ping=AsyncMock())
        healthy = {"status": "ready", "checks": {"worker": {"status": "ok"}}}
        with (
            patch.object(main.app.state, "checkpointer", saver, create=True),
            patch.object(main.app.state, "heartbeat_task", SimpleNamespace(done=lambda: False), create=True),
            patch.object(main, "readiness_check", return_value=healthy),
            patch.object(main, "material_worker_running", return_value=True),
            patch.object(main, "report_worker_running", return_value=False),
            patch.object(main.interview, "generation_worker_running", return_value=True),
            patch.object(main.session, "chat_worker_running", return_value=True),
        ):
            response = await main.health_ready(object())

        self.assertEqual(response.status_code, 503)
        self.assertEqual(healthy["checks"]["report_worker"]["error_code"], "worker_unavailable")
        self.assertEqual(healthy["checks"]["material_worker"]["status"], "ok")

    async def test_material_shutdown_clears_task_liveness(self) -> None:
        """The material poller is no longer advertised once shutdown cancels it."""
        import asyncio

        from app.services.materials import knowledge

        async def wait_forever() -> None:
            await asyncio.Event().wait()

        task = asyncio.create_task(wait_forever())
        with patch.object(knowledge, "_material_worker", task):
            self.assertTrue(knowledge.material_worker_running())
            await knowledge.stop_material_worker()
            self.assertFalse(knowledge.material_worker_running())
        self.assertTrue(task.cancelled())


if __name__ == "__main__":
    unittest.main()
