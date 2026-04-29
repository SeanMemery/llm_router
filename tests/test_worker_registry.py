from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from worker_registry import (
    LLMWorkerHeartbeatRequest,
    LLMWorkerRecord,
    LLMWorkerRegisterRequest,
    LLMWorkerRegistry,
    LLMWorkerStatus,
    LLMWorkerTransportMode,
)


class WorkerRegistryTests(unittest.TestCase):
    def test_register_creates_online_worker_and_connection(self) -> None:
        registry = LLMWorkerRegistry()
        worker = registry.register(
            LLMWorkerRegisterRequest(
                display_name="gpu-1",
                hostname="gpu-1",
                endpoint_base_url="http://worker:8080",
                model="qwen-gguf",
                local_model_root="/models",
                max_concurrent_requests=2,
            )
        )

        self.assertEqual(worker.status, LLMWorkerStatus.online)
        self.assertEqual(worker.max_concurrent_requests, 2)
        configs = registry.active_connection_configs()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].source_kind, "worker")
        self.assertEqual(configs[0].worker_id, worker.worker_id)

    def test_heartbeat_updates_worker_fields(self) -> None:
        registry = LLMWorkerRegistry()
        worker = registry.register(
            LLMWorkerRegisterRequest(
                display_name="gpu-1",
                hostname="gpu-1",
                endpoint_base_url="http://worker:8080",
                model="model-a",
            )
        )

        updated = registry.heartbeat(
            worker.worker_id,
            LLMWorkerHeartbeatRequest(
                model="model-b",
                endpoint_base_url="http://worker:8081",
                last_error="temporary",
            ),
        )

        self.assertEqual(updated.model, "model-b")
        self.assertEqual(updated.endpoint_base_url, "http://worker:8081")
        self.assertEqual(updated.last_error, "temporary")

    def test_router_parallel_limit_clamps_effective_connection_capacity(self) -> None:
        registry = LLMWorkerRegistry()
        worker = registry.register(
            LLMWorkerRegisterRequest(
                display_name="gpu-1",
                hostname="gpu-1",
                endpoint_base_url=None,
                model="model-a",
                transport_mode=LLMWorkerTransportMode.pull_request,
                max_concurrent_requests=6,
            )
        )

        updated = registry.set_router_max_concurrent_requests(
            worker.worker_id,
            router_max_concurrent_requests=3,
        )

        self.assertEqual(updated.router_max_concurrent_requests, 3)
        self.assertEqual(updated.effective_max_concurrent_requests, 3)
        configs = registry.active_connection_configs()
        self.assertEqual(configs[0].max_concurrent_requests, 3)

    def test_register_preserves_router_parallel_limit_across_reregistration(self) -> None:
        registry = LLMWorkerRegistry()
        worker = registry.register(
            LLMWorkerRegisterRequest(
                worker_id="worker-1",
                display_name="gpu-1",
                hostname="gpu-1",
                endpoint_base_url=None,
                model="model-a",
                transport_mode=LLMWorkerTransportMode.pull_request,
                max_concurrent_requests=6,
            )
        )
        registry.set_router_max_concurrent_requests(
            worker.worker_id,
            router_max_concurrent_requests=2,
        )

        updated = registry.register(
            LLMWorkerRegisterRequest(
                worker_id="worker-1",
                display_name="gpu-1",
                hostname="gpu-1",
                endpoint_base_url=None,
                model="model-a",
                transport_mode=LLMWorkerTransportMode.pull_request,
                max_concurrent_requests=5,
            )
        )

        self.assertEqual(updated.max_concurrent_requests, 5)
        self.assertEqual(updated.router_max_concurrent_requests, 2)
        self.assertEqual(updated.effective_max_concurrent_requests, 2)

    def test_register_preserves_router_priority_across_reregistration(self) -> None:
        registry = LLMWorkerRegistry()
        worker = registry.register(
            LLMWorkerRegisterRequest(
                worker_id="worker-1",
                display_name="gpu-1",
                hostname="gpu-1",
                endpoint_base_url=None,
                model="model-a",
                transport_mode=LLMWorkerTransportMode.pull_request,
                max_concurrent_requests=6,
            )
        )
        registry.set_router_priority(
            worker.worker_id,
            router_priority=1,
        )

        updated = registry.register(
            LLMWorkerRegisterRequest(
                worker_id="worker-1",
                display_name="gpu-1",
                hostname="gpu-1",
                endpoint_base_url=None,
                model="model-a",
                transport_mode=LLMWorkerTransportMode.pull_request,
                max_concurrent_requests=5,
            )
        )

        self.assertEqual(updated.router_priority, 1)
        configs = registry.active_connection_configs()
        self.assertEqual(configs[0].priority, 1)

    def test_prune_marks_stale_worker_offline(self) -> None:
        stale_time = datetime.now(UTC) - timedelta(seconds=120)
        registry = LLMWorkerRegistry(
            workers=[
                LLMWorkerRecord(
                    worker_id="worker-1",
                    display_name="gpu-1",
                    hostname="gpu-1",
                    endpoint_base_url="http://worker:8080",
                    model="model-a",
                    max_concurrent_requests=1,
                    registered_at=stale_time,
                    last_heartbeat_at=stale_time,
                )
            ],
            heartbeat_timeout_seconds=30,
        )

        stale = registry.prune_stale_workers()

        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0].status, LLMWorkerStatus.offline)
        self.assertEqual(registry.active_connection_configs(), [])

    def test_pull_request_worker_can_register_without_endpoint_url(self) -> None:
        registry = LLMWorkerRegistry()
        worker = registry.register(
            LLMWorkerRegisterRequest(
                display_name="gpu-1",
                hostname="gpu-1",
                endpoint_base_url=None,
                model="qwen-gguf",
                transport_mode=LLMWorkerTransportMode.pull_request,
                max_concurrent_requests=2,
            )
        )

        self.assertEqual(worker.transport_mode, LLMWorkerTransportMode.pull_request)
        self.assertIsNone(worker.endpoint_base_url)
        configs = registry.active_connection_configs()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].transport_mode, "pull-request")
        self.assertEqual(configs[0].base_url, f"worker://{worker.worker_id}")
        self.assertFalse(configs[0].starred)

    def test_workers_and_active_connections_are_sorted_by_priority_then_display_name(self) -> None:
        registry = LLMWorkerRegistry()
        zeta = registry.register(
            LLMWorkerRegisterRequest(
                display_name="zeta-worker",
                hostname="gpu-z",
                endpoint_base_url=None,
                model="model-z",
                transport_mode=LLMWorkerTransportMode.pull_request,
            )
        )
        registry.set_router_priority(zeta.worker_id, router_priority=3)
        alpha = registry.register(
            LLMWorkerRegisterRequest(
                display_name="alpha-worker",
                hostname="gpu-a",
                endpoint_base_url=None,
                model="model-a",
                transport_mode=LLMWorkerTransportMode.pull_request,
            )
        )
        registry.set_router_priority(alpha.worker_id, router_priority=1)

        workers = registry.list_workers()
        self.assertEqual([item.display_name for item in workers], ["alpha-worker", "zeta-worker"])

        configs = registry.active_connection_configs()
        self.assertEqual([item.worker_id for item in configs], [item.worker_id for item in workers])


if __name__ == "__main__":
    unittest.main()
