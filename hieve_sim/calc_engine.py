from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

from .utils import LOGGER


@dataclass
class ChannelState:
    ch_id: int
    weight: int
    bytetrack: object
    fps: float = 30.0
    num_frames: int = 0
    last_callback_ms: Optional[int] = None


class CallbackContext:
    __slots__ = ("_engine", "_channel", "_now_ms", "_prev_ms")

    def __init__(self, engine: "AISimEngine", channel: ChannelState, now_ms: int, prev_ms: Optional[int]):
        self._engine = engine
        self._channel = channel
        self._now_ms = now_ms
        self._prev_ms = prev_ms

    @property
    def ch(self) -> int:
        return self._channel.ch_id

    @property
    def now_ms(self) -> int:
        return self._now_ms

    @property
    def prev_ms(self) -> Optional[int]:
        return self._prev_ms

    @property
    def bytetrack(self):
        return self._channel.bytetrack

    @property
    def weight(self) -> int:
        return self._channel.weight

    @property
    def total_weight(self) -> int:
        return self._engine.total_weight

    # Optional: allow changing weights (not used in this project by default)
    def boost_weight(self) -> int:
        return self._engine._set_weight(self._channel.ch_id, self._channel.weight * 4)

    def decay_weight(self) -> int:
        return self._engine._set_weight(self._channel.ch_id, max(1, self._channel.weight // 2))


class AISimEngine:
    """AI compute simulation engine (multi-channel weighted scheduling).

    - Each tick selects one channel using random weighted scheduling (probability proportional to weight).
    - Calls callback(ctx).
    - Time unit: ms.

    Notes:
    - For this HIEVE simulation, weights are typically equal and unchanged.
    - Provides both threaded (start/stop) and blocking (run_blocking) modes.
    """

    def __init__(
        self,
        channel_bytetracks: List[object],
        channel_weights: Optional[List[int]] = None,
        channel_fps: Optional[List[float]] = None,
        channel_num_frames: Optional[List[int]] = None,
    ):
        if not channel_bytetracks:
            raise ValueError("channel_bytetracks must not be empty")

        if channel_weights is None:
            channel_weights = [1] * len(channel_bytetracks)
        if len(channel_weights) != len(channel_bytetracks):
            raise ValueError("channel_weights length mismatch")

        if channel_fps is not None and len(channel_fps) != len(channel_bytetracks):
            raise ValueError("channel_fps length mismatch")
        if channel_num_frames is not None and len(channel_num_frames) != len(channel_bytetracks):
            raise ValueError("channel_num_frames length mismatch")

        self._lock = threading.RLock()
        self._channels: List[ChannelState] = []
        for i, (w, bt) in enumerate(zip(channel_weights, channel_bytetracks)):
            w = int(w)
            if w < 1 or w > 4:
                raise ValueError(f"Initial weight out of range [1,4]: ch={i}, weight={w}")
            fps = float(channel_fps[i]) if channel_fps is not None else 30.0
            num_frames = int(channel_num_frames[i]) if channel_num_frames is not None else 0
            self._channels.append(ChannelState(ch_id=i, weight=w, bytetrack=bt, fps=fps, num_frames=num_frames))

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Callback return:
        #   - bool: progressed or not (legacy)
        #   - int:  0=no progress, 1=progress (normal cost), 2=progress (double cost), ...
        self._callback: Optional[Callable[[CallbackContext], Union[bool, int]]] = None

        # Weighted random schedule table
        self._rebuild_schedule_locked()

    @property
    def total_weight(self) -> int:
        with self._lock:
            return sum(ch.weight for ch in self._channels)

    def get_weight(self, ch_id: int) -> int:
        with self._lock:
            return self._channels[ch_id].weight

    def _set_weight(self, ch_id: int, new_weight: int) -> int:
        with self._lock:
            new_weight = int(new_weight)
            if new_weight < 1:
                new_weight = 1
            elif new_weight > 4:
                new_weight = 4
            self._channels[ch_id].weight = new_weight
            # keep schedule consistent with weights
            self._rebuild_schedule_locked()
            return new_weight

    @property
    def max_weight(self) -> int:
        # Keep consistent with the clamp in _set_weight
        return 4

    def _rebuild_schedule_locked(self) -> None:
        """Build a weighted schedule table (each channel repeated by weight).

        This table is used for random weighted selection.
        If weights change, we rebuild the table.
        """
        self._schedule: List[ChannelState] = []
        for ch in self._channels:
            self._schedule.extend([ch] * int(ch.weight))
        if not self._schedule:
            self._schedule = list(self._channels)
        self._schedule_pos: int = 0

    def _choose_channel(self) -> ChannelState:
        # Random weighted scheduling:
        # P(select ch_i) = weight_i / sum_j weight_j
        with self._lock:
            if not hasattr(self, "_schedule") or not getattr(self, "_schedule", None):
                self._rebuild_schedule_locked()
            return random.choice(self._schedule)


    def set_callback(self, callback: Callable[[CallbackContext], Union[bool, int]]) -> None:
        self._callback = callback

    def start(self, *, tick_ms: int = 0, total_ms: Optional[int] = None, virtual_time: bool = True) -> None:
        if self._callback is None:
            raise RuntimeError("callback not set. call set_callback() first.")
        if self._thread and self._thread.is_alive():
            raise RuntimeError("engine already running")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, args=(tick_ms, total_ms, virtual_time), daemon=True
        )
        self._thread.start()

    def stop(self, *, join: bool = True, timeout: Optional[float] = None) -> None:
        self._stop_event.set()
        if join and self._thread:
            self._thread.join(timeout=timeout)

    def run_blocking(self, *, tick_ms: int, total_ms: int, virtual_time: bool = True, verbose: bool = False):
        """Run loop in current thread (handy for CLI/testing)."""
        if self._callback is None:
            raise RuntimeError("callback not set. call set_callback() first.")
        return self._run_loop(tick_ms, total_ms, virtual_time, verbose=verbose)

    def _run_loop(self, tick_ms: int, total_ms: Optional[int], virtual_time: bool, verbose: bool = False):
        """Core scheduling loop.

        Virtual-time mode (virtual_time=True) implements *time-stable scheduling*:
        - now_ms stays constant while we try different channels.
        - Each channel is visited at most once for the current now_ms.
        - If a chosen channel cannot advance (e.g., same frame), we do NOT advance time; we try another channel.
        - Only when no channels remain eligible at this now_ms do we advance now_ms by tick_ms.

        This matches the project's requirement: if frame doesn't change for a channel, tick_ms must not accumulate.
        """
        callback = self._callback
        assert callback is not None

        if tick_ms < 0:
            tick_ms = 0

        now_ms = 0 if virtual_time else int(time.time() * 1000)
        start_wall = time.time()

        if virtual_time:

            tick_count = 0

            while not self._stop_event.is_set():
                if total_ms is not None and now_ms > total_ms:
                    break

                # Pick among channels not yet tried at this now_ms.
                with self._lock:
                    schedule = list(self._schedule) if getattr(self, "_schedule", None) else list(self._channels)

                progressed = False
                progressed_cost = 0
                
                blocked: set[int] = set()  # channels already tried at this now_ms

                _frame_idx = lambda ms, fps: int((ms * fps) / 1000.0)  # cheap ms->frame mapping


                while not progressed:
                    # Pre-filter channels that cannot advance (same frame) or are already past end.
                    candidates = []
                    for ch in schedule:
                        if ch.ch_id in blocked:
                            continue
                        now_frame = _frame_idx(now_ms, ch.fps)
                        if ch.num_frames > 0 and now_frame >= ch.num_frames:
                            continue
                        prev_ms = ch.last_callback_ms
                        prev_frame = _frame_idx(prev_ms, ch.fps) if prev_ms is not None else -1
                        if now_frame == prev_frame:
                            continue
                        candidates.append(ch)
                    if not candidates:
                        break
                    ch = random.choice(candidates)
                    prev_ms = ch.last_callback_ms
                    ctx = CallbackContext(self, ch, now_ms, prev_ms)
                    t0 = time.perf_counter()
                    try:
                        # Support both legacy bool and integer "cost units" return.
                        # bool is a subclass of int, so int(True)=1, int(False)=0.
                        cost_units = int(callback(ctx) or 0)
                        progressed = cost_units > 0
                        progressed_cost = cost_units if progressed else 0
                    except Exception as e:
                        LOGGER.warning(f"callback error on ch={ch.ch_id}: {e}")
                    t1 = time.perf_counter()
                    exec_ms = int((t1 - t0) * 1000)
                    # Mark this channel as visited at this now_ms regardless of progress.
                    ch.last_callback_ms = now_ms
                    blocked.add(ch.ch_id)

                    if verbose:
                        LOGGER.info(
                            f"[TASK] now_ms={now_ms} ch={ch.ch_id} progressed={int(progressed)} exec_ms={exec_ms} "
                            f"weight={ch.weight} totalW={self.total_weight}"
                        )

                # Compute accounting:
                # - Default is 1 tick per progressed callback.
                # - If callback returns cost_units>1 (e.g. retro-fill reads 2 DETs),
                #   we advance virtual time by tick_ms * cost_units.
                if progressed and progressed_cost > 0:
                    now_ms += tick_ms * int(progressed_cost)
                    tick_count += int(progressed_cost)
                else:
                    now_ms += tick_ms
                    tick_count += 1
                if not verbose and (tick_count % 250 == 0):
                    # Heartbeat to keep long simulations observable
                    LOGGER.info(f"[PROGRESS] now_ms={now_ms}/{total_ms if total_ms is not None else -1} tick={tick_count}")
            return

        # Real-time mode (kept close to original behavior)
        while not self._stop_event.is_set():
            if total_ms is not None and (not virtual_time) and (int(time.time() * 1000) - now_ms) > total_ms:
                break

            # choose channel
            ch = self._choose_channel()
            prev_ms = ch.last_callback_ms
            ctx = CallbackContext(self, ch, now_ms, prev_ms)

            t0 = time.perf_counter()
            try:
                callback(ctx)
            except Exception as e:
                LOGGER.warning(f"callback error on ch={ch.ch_id}: {e}")
            t1 = time.perf_counter()
            exec_ms = int((t1 - t0) * 1000)

            ch.last_callback_ms = now_ms

            if verbose:
                LOGGER.info(f"[TASK] ch={ch.ch_id} exec_ms={exec_ms} weight={ch.weight} totalW={self.total_weight}")

            # real time: sleep to honor tick_ms
            if tick_ms > 0:
                used_ms = int((time.time() - start_wall) * 1000) - (now_ms - int(time.time() * 1000))
                sleep_ms = max(0, tick_ms - used_ms)
                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000.0)
            now_ms = int(time.time() * 1000)

        return
