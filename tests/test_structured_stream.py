"""
Tests for ingestion/structured_stream.py's WebSocket reconnect handling.

Real incident (2026-09-02): a dropped Binance WebSocket read loop
(binance.exceptions.ReadLoopClosed) propagated all the way out of
StructuredStream.start() and killed the whole crypto process with no
recovery. These tests verify the fix — start() now catches the
reconnectable exceptions and retries with backoff instead of exiting —
without touching a real network connection: AsyncClient.create and
BinanceSocketManager are mocked, and asyncio.sleep is patched out so the
backoff delays don't actually elapse.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from binance.exceptions import ReadLoopClosed
from ingestion.structured_stream import StructuredStream


class _StopTest(Exception):
    """Sentinel raised once the reconnect loop has proven itself — lets a
    test assert "it kept going" without an actually-infinite while True."""


def _make_socket_manager(behaviors: list):
    """
    Each call to multiplex_socket() pops the next behavior:
      - an Exception instance/class: the `async with` block raises it immediately
      - a list of messages: yields each via recv(), then returns None (clean end)
    """
    behaviors = list(behaviors)

    @asynccontextmanager
    async def multiplex_socket(streams):
        behavior = behaviors.pop(0)
        if isinstance(behavior, type) and issubclass(behavior, Exception):
            raise behavior("simulated drop")
        if isinstance(behavior, Exception):
            raise behavior

        msock = MagicMock()
        messages = list(behavior) + [None]

        async def recv():
            return messages.pop(0)

        msock.recv = recv
        yield msock

    bm = MagicMock()
    bm.multiplex_socket = multiplex_socket
    return bm


def _run_start_until_stopped(stream: StructuredStream, socket_manager) -> None:
    async def _drive():
        with patch("ingestion.structured_stream.AsyncClient.create", new=AsyncMock(return_value=MagicMock())), \
             patch("ingestion.structured_stream.BinanceSocketManager", return_value=socket_manager), \
             patch("ingestion.structured_stream.asyncio.sleep", new=AsyncMock()):
            await stream.start()

    with pytest.raises(_StopTest):
        asyncio.run(_drive())


def test_read_loop_closed_does_not_crash_the_stream() -> None:
    """The exact real-incident exception (ReadLoopClosed) must be caught
    and retried, not propagate out of start()."""
    stream = StructuredStream()
    socket_manager = _make_socket_manager([ReadLoopClosed, _StopTest])
    _run_start_until_stopped(stream, socket_manager)
    # Reaching _StopTest on the second connection attempt proves the first
    # ReadLoopClosed was swallowed and reconnected, not left to propagate.


def test_clean_disconnect_also_reconnects() -> None:
    """A clean end (recv() returns None, no exception) must reconnect too --
    not just the exception path."""
    stream = StructuredStream()
    socket_manager = _make_socket_manager([[], _StopTest])  # no messages -> immediate None
    _run_start_until_stopped(stream, socket_manager)


def test_messages_are_still_handled_after_a_reconnect() -> None:
    """A message delivered on the connection AFTER a drop must still reach
    _handle — reconnecting shouldn't silently drop the stream's data flow."""
    stream = StructuredStream()
    received = []
    stream.add_callback(lambda sym, ind: received.append((sym, ind)))

    msg = {
        "data": {
            "s": next(iter(stream._buffers)),
            "k": {"i": "5m", "c": "100.0", "v": "10", "o": "99", "h": "101", "l": "98", "x": False},
        }
    }
    socket_manager = _make_socket_manager([ReadLoopClosed, [msg], _StopTest])
    _run_start_until_stopped(stream, socket_manager)

    assert len(received) == 1
    assert received[0][0] == next(iter(stream._buffers))
