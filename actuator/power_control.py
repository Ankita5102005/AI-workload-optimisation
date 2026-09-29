"""The ONLY module allowed to read, set or restore a GPU's power limit.

Every other part of this codebase that needs the power limit must go through
here. That is a deliberate constraint from the project plan: keeping this in
one place is what makes "restore the original limit no matter what happens"
possible to actually guarantee.

Setting a power limit requires root (`nvidia-smi -i <idx> -pl <watts>`), and
NVML mirrors that: `nvmlDeviceSetPowerManagementLimit` needs the same
privilege. Reading never does.

Normal use is the context manager, which restores on ANY exit path,
including an exception, Ctrl+C, or SIGTERM:

    from actuator.power_control import PowerLimitSession

    with PowerLimitSession(device_index=0, expect_uuid=UUID) as pl:
        pl.set_w(250)
        ... run one measurement ...
    # power limit is back to whatever it was when the `with` block was entered,
    # even if the code inside raised.

For the coarser "protect the whole session, survive even a killed process"
guarantee, see scripts/session_guard.sh, which wraps a session at the shell
level using the state file this module writes.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import pynvml
except ImportError:
    pynvml = None

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = Path("/raid/gpu_profiler/power_state")  # override via PowerLimitSession(state_dir=...)


class PowerControlError(RuntimeError):
    pass


@dataclass
class PowerLimitState:
    gpu_uuid: str
    device_index: int
    original_limit_w: float
    min_limit_w: float
    max_limit_w: float
    default_limit_w: float
    recorded_at: float

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "PowerLimitState":
        return cls(**json.loads(text))


def _require_pynvml() -> None:
    if pynvml is None:
        raise PowerControlError("pynvml is not installed. Run: pip install nvidia-ml-py")


def read_limit_w(device_index: int) -> float:
    """Current power limit, in watts. Never requires elevated privilege."""
    _require_pynvml()
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
    finally:
        pynvml.nvmlShutdown()


def read_full_state(device_index: int) -> PowerLimitState:
    _require_pynvml()
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        uuid = pynvml.nvmlDeviceGetUUID(h)
        uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
        lo, hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
        return PowerLimitState(
            gpu_uuid=uuid, device_index=device_index,
            original_limit_w=pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0,
            min_limit_w=lo / 1000.0, max_limit_w=hi / 1000.0,
            default_limit_w=pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000.0,
            recorded_at=time.time(),
        )
    finally:
        pynvml.nvmlShutdown()


def set_limit_w(device_index: int, watts: float, *, expect_uuid: Optional[str] = None) -> None:
    """Set the power limit. Requires root. Refuses if expect_uuid is given and doesn't match,
    so a stale device index can never silently hit the wrong physical card."""
    _require_pynvml()
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        if expect_uuid is not None:
            uuid = pynvml.nvmlDeviceGetUUID(h)
            uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
            if uuid != expect_uuid:
                raise PowerControlError(
                    f"Refusing to set power limit: device index {device_index} has UUID "
                    f"{uuid}, expected {expect_uuid}. The device numbering may have changed."
                )
        lo, hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
        lo, hi = lo / 1000.0, hi / 1000.0
        if not (lo <= watts <= hi):
            raise PowerControlError(f"{watts} W is outside the supported range [{lo}, {hi}] W")
        try:
            pynvml.nvmlDeviceSetPowerManagementLimit(h, int(watts * 1000))
        except pynvml.NVMLError_NoPermission as e:
            raise PowerControlError(
                "NVML refused to set the power limit (no permission). This needs root."
            ) from e
        logger.info("GPU %d power limit set to %.0f W", device_index, watts)
    finally:
        pynvml.nvmlShutdown()


class PowerLimitSession(AbstractContextManager):
    """Context manager: records the power limit on entry, restores it on any exit.

    Also writes a small JSON state file (device UUID, original limit, timestamp) so a
    coarser shell-level wrapper (scripts/session_guard.sh) can detect and fix an
    unrestored limit even if this Python process is killed outright and never reaches
    its own __exit__ (e.g. `kill -9`, an OOM kill, a lost SSH connection with no
    surviving shell). That outer layer is the second line of defence; this class is
    the first and should be sufficient for every case except an unclean kill.
    """

    def __init__(self, device_index: int, *, expect_uuid: Optional[str] = None,
                 state_dir: Path = DEFAULT_STATE_DIR):
        self.device_index = device_index
        self.expect_uuid = expect_uuid
        self.state_dir = Path(state_dir)
        self.state: Optional[PowerLimitState] = None
        self._state_path: Optional[Path] = None
        self._prev_handlers = {}

    def __enter__(self) -> "PowerLimitSession":
        self.state = read_full_state(self.device_index)
        if self.expect_uuid is not None and self.state.gpu_uuid != self.expect_uuid:
            raise PowerControlError(
                f"Refusing to start: device index {self.device_index} has UUID "
                f"{self.state.gpu_uuid}, expected {self.expect_uuid}."
            )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.state_dir / f"{self.state.gpu_uuid}.json"
        if self._state_path.exists():
            raise PowerControlError(
                f"A power-limit session state file already exists for GPU {self.state.gpu_uuid} "
                f"({self._state_path}). Either another session is active on this GPU, or a "
                "previous session crashed before cleaning up — run scripts/reconcile_power.py "
                "first to check and restore it, then delete the file if it's confirmed stale."
            )
        self._state_path.write_text(self.state.to_json())
        logger.info("Session started: GPU %d (%s) at %.0f W (default %.0f W)",
                   self.device_index, self.state.gpu_uuid, self.state.original_limit_w,
                   self.state.default_limit_w)
        # Best-effort: also restore on SIGTERM (e.g. `kill`, not `kill -9`), not just
        # normal exceptions. SIGINT (Ctrl+C) already raises KeyboardInterrupt, which
        # __exit__ below catches via the normal `with` protocol.
        def _on_sigterm(signum, frame):
            self._restore()
            raise SystemExit(f"Terminated by signal {signum}; power limit restored.")
        self._prev_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, _on_sigterm)
        return self

    def set_w(self, watts: float) -> None:
        set_limit_w(self.device_index, watts, expect_uuid=self.state.gpu_uuid)

    def current_w(self) -> float:
        return read_limit_w(self.device_index)

    def _restore(self) -> None:
        if self.state is None:
            return
        current = read_limit_w(self.device_index)
        if abs(current - self.state.original_limit_w) > 0.5:
            set_limit_w(self.device_index, self.state.original_limit_w, expect_uuid=self.state.gpu_uuid)
            logger.info("Restored GPU %d to %.0f W", self.device_index, self.state.original_limit_w)
        if self._state_path is not None and self._state_path.exists():
            self._state_path.unlink()

    def __exit__(self, exc_type, exc, tb) -> bool:
        for sig, handler in self._prev_handlers.items():
            signal.signal(sig, handler)
        self._restore()
        return False  # never swallow the original exception
