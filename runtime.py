"""Connects Bulk to Hermes services at call time.

Hermes modules are imported inside functions because ``hermes plugins validate`` loads
the plugin in a bare interpreter. Managers are keyed by data directory because
``plugin_data_dir`` follows the active Hermes profile.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .jobs import JobManager
from .store import Store

PLUGIN_NAME = "bulk"
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_CONCURRENCY = 4
# A job with no --max-tokens still stops here, so a mistyped path or instruction cannot run up an open-ended bill.
DEFAULT_MAX_TOKENS = 200_000
_STOPPING_STATES = ("draining", "stopped")


def _hermes_data_dir() -> Path:
    from plugins.plugin_storage import plugin_data_dir

    return plugin_data_dir(PLUGIN_NAME)


def _hermes_open_db() -> sqlite3.Connection:
    from plugins.plugin_storage import plugin_db

    return plugin_db(PLUGIN_NAME)


def _hermes_cwd() -> Path:
    from agent.runtime_cwd import resolve_agent_cwd

    return resolve_agent_cwd()


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path("~/.hermes").expanduser()


def _hermes_read_block(path: str) -> Optional[str]:
    """The host's own read deny list (credential stores, project env files, device files), when it provides one."""
    try:
        from agent.file_safety import get_read_block_error
    except Exception:
        return None
    return get_read_block_error(path)


def _hermes_gateway_status() -> Any:
    from gateway.status import read_runtime_status

    return read_runtime_status()


def gateway_is_stopping(read_status: Callable[[], Any] = _hermes_gateway_status) -> bool:
    """True once the gateway running this process has started to shut down.

    Hermes gives plugins no shutdown callback: the gateway exits without calling
    ``on_unload``. It does publish its state through ``gateway.status``. Anything
    unexpected counts as "not stopping", and the claim is simply recovered after the
    restart. The pid check keeps a separate CLI process from reacting to the gateway's
    shutdown.
    """
    try:
        state = read_status()
    except Exception:
        return False
    return (
        isinstance(state, dict)
        and state.get("pid") == os.getpid()
        and state.get("gateway_state") in _STOPPING_STATES
    )


class Runtime:
    def __init__(
        self,
        ctx: Any,
        data_dir: Callable[[], Path] = _hermes_data_dir,
        open_db: Callable[[], sqlite3.Connection] = _hermes_open_db,
        cwd: Callable[[], Path] = _hermes_cwd,
        sleep: Callable[[float], None] = time.sleep,
        hermes_home: Callable[[], Path] = _hermes_home,
        read_block: Callable[[str], Optional[str]] = _hermes_read_block,
    ) -> None:
        self._ctx = ctx
        self._data_dir = data_dir
        self._open_db = open_db
        self._sleep = sleep
        self.cwd = cwd
        self.hermes_home = hermes_home
        self.read_block = read_block
        self._lock = threading.Lock()
        self._managers: dict[Path, JobManager] = {}

    def manager(self) -> JobManager:
        data_dir = self._data_dir()
        with self._lock:
            manager: Optional[JobManager] = self._managers.get(data_dir)
            if manager is None:
                manager = JobManager(
                    Store(self._open_db()),
                    data_dir,
                    self._ctx.llm,
                    self._sleep,
                    draining=gateway_is_stopping,
                )
                self._managers[data_dir] = manager
            return manager

    def default_retries(self) -> int:
        """The ``max_retries`` plugin setting, or the default when it is missing or invalid."""
        return self._int_setting("max_retries", DEFAULT_MAX_RETRIES, minimum=0)

    def max_concurrency(self) -> int:
        """The ``max_concurrency`` plugin setting: the most items a job may run at once."""
        return self._int_setting("max_concurrency", DEFAULT_MAX_CONCURRENCY, minimum=1)

    def default_max_tokens(self) -> int:
        """The ``default_max_tokens`` plugin setting: the token limit of a job created without --max-tokens."""
        return self._int_setting("default_max_tokens", DEFAULT_MAX_TOKENS, minimum=1)

    def allowed_roots(self) -> list[Path]:
        """Folders ``--over`` may name, most preferred first (a relative path is resolved against the first).

        The ``allowed_roots`` plugin setting, else the agent's working directory and ``<HERMES_HOME>/workspace``.
        A folder that is the Hermes home or contains it is never allowed: the home holds credentials, env files and
        every other profile, and a job would send whatever it reads to the model.
        """
        configured = self._ctx.get_config("allowed_roots", None)
        if isinstance(configured, str):
            configured = [configured]
        home = self.hermes_home().expanduser().resolve()
        if configured:
            candidates = [Path(str(value)).expanduser() for value in configured if str(value).strip()]
            candidates = [path for path in candidates if path.is_absolute()]
        else:
            candidates = [self.cwd(), home / "workspace"]
        roots: list[Path] = []
        for candidate in candidates:
            root = candidate.resolve()
            if home.is_relative_to(root) or not root.is_dir() or root in roots:
                continue
            roots.append(root)
        return roots

    def _int_setting(self, key: str, default: int, minimum: int) -> int:
        try:
            value = int(self._ctx.get_config(key, default))
        except (TypeError, ValueError):
            return default
        return value if value >= minimum else default

    def close(self) -> None:
        with self._lock:
            for manager in self._managers.values():
                manager.close()
            self._managers.clear()
