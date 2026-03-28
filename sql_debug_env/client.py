"""
HTTP client for the SQL Debug & Optimize RL Environment.

Follows the official OpenEnv client interface:
  result = await env.reset()          -> StepResult
  result = await env.step(action)     -> StepResult
  result.observation                  -> SQLDebugObservation
  result.reward                       -> float
  result.done                         -> bool
  await env.close()

Also supports sync usage and from_docker_image() factory.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from typing import Optional

import httpx

from .models import SQLDebugAction, SQLDebugObservation, SQLDebugState, StepResult


class SQLDebugEnv:
    """
    HTTP client for the SQL Debug & Optimize RL Environment.

    Supports both async and sync usage, plus a from_docker_image() factory
    that spins up a local Docker container automatically.
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._container_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "SQLDebugEnv":
        self._async_client = httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout
        )
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Sync context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "SQLDebugEnv":
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self

    def __exit__(self, *args) -> None:
        if self._client:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Factory: spin up a local Docker container
    # ------------------------------------------------------------------

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        host_port: int = 7860,
        env_vars: Optional[dict] = None,
        wait_seconds: int = 15,
    ) -> "SQLDebugEnv":
        """
        Pull & run a Docker image, wait for it to be healthy, return a connected client.
        Mirrors the OpenEnv from_docker_image() pattern.

        Usage:
            IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "sql-debug-env:latest")
            env = await SQLDebugEnv.from_docker_image(IMAGE_NAME)
        """
        extra_env = env_vars or {}
        passthrough = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
        env_flags = []
        for var in passthrough:
            val = os.environ.get(var) or extra_env.get(var)
            if val:
                env_flags += ["-e", f"{var}={val}"]
        for k, v in extra_env.items():
            if k not in passthrough:
                env_flags += ["-e", f"{k}={v}"]

        cmd = [
            "docker", "run", "-d",
            "-p", f"{host_port}:7860",
            *env_flags,
            image_name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr.strip()}")

        container_id = result.stdout.strip()
        base_url = f"http://localhost:{host_port}"
        instance = cls(base_url=base_url)
        instance._container_id = container_id
        instance._async_client = httpx.AsyncClient(
            base_url=base_url, timeout=instance.timeout
        )

        # Wait for health check
        for _ in range(wait_seconds * 2):
            try:
                r = await instance._async_client.get("/health")
                if r.status_code == 200:
                    return instance
            except Exception:
                pass
            await asyncio.sleep(0.5)

        raise RuntimeError(
            f"Container {container_id[:12]} did not become healthy in {wait_seconds}s"
        )

    # ------------------------------------------------------------------
    # Async OpenEnv API
    # ------------------------------------------------------------------

    async def reset(self, seed: Optional[int] = None) -> StepResult:
        """Start a new episode. Returns StepResult with .observation, .reward=0, .done=False."""
        data = await self._async_post("/reset", {"seed": seed})
        obs = SQLDebugObservation(**data)
        return StepResult(observation=obs, reward=0.0, done=False)

    async def step(self, action: SQLDebugAction) -> StepResult:
        """Submit fixed_query. Returns StepResult with .observation, .reward, .done."""
        data = await self._async_post(
            "/step",
            {"fixed_query": action.fixed_query, "explanation": action.explanation},
        )
        obs = SQLDebugObservation(**data)
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    async def get_state(self) -> SQLDebugState:
        r = await self._get_async_client().get("/state")
        r.raise_for_status()
        return SQLDebugState(**r.json())

    async def close(self) -> None:
        """Close connections and stop Docker container if started via from_docker_image."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
        if self._container_id:
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
            )
            self._container_id = None

    # ------------------------------------------------------------------
    # Sync convenience wrappers
    # ------------------------------------------------------------------

    def reset_sync(self, seed: Optional[int] = None) -> StepResult:
        data = self._sync_post("/reset", {"seed": seed})
        obs = SQLDebugObservation(**data)
        return StepResult(observation=obs, reward=0.0, done=False)

    def step_sync(self, action: SQLDebugAction) -> StepResult:
        data = self._sync_post(
            "/step",
            {"fixed_query": action.fixed_query, "explanation": action.explanation},
        )
        obs = SQLDebugObservation(**data)
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    def health(self) -> dict:
        r = self._get_sync_client().get("/health")
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_sync_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout
            )
        return self._async_client

    def _sync_post(self, path: str, payload: dict) -> dict:
        r = self._get_sync_client().post(path, json=payload)
        r.raise_for_status()
        return r.json()

    async def _async_post(self, path: str, payload: dict) -> dict:
        r = await self._get_async_client().post(path, json=payload)
        r.raise_for_status()
        return r.json()
