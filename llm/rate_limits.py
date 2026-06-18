from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from threading import Lock

SLOW_REQUEST_SECONDS = 60.0


def is_rate_limit_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "rate_limit" in name:
        return True

    status = getattr(exc, "status_code", None)
    if status == 429:
        return True

    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True

    message = str(exc).lower()
    return (
        "rate limit" in message
        or "rate_limit" in message
        or "too many requests" in message
        or "429" in message
    )


@dataclass
class RateLimitMonitor:
    rate_limit_hits: int = 0
    slow_requests: int = 0
    events: list[dict] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)
    _last_alert_at: float = field(default=0.0, repr=False)

    def reset(self) -> None:
        with self._lock:
            self.rate_limit_hits = 0
            self.slow_requests = 0
            self.events = []
            self._last_alert_at = 0.0

    def record_hit(self, provider: str, model: str, exc: BaseException) -> None:
        with self._lock:
            self.rate_limit_hits += 1
            count = self.rate_limit_hits
            self.events.append(
                {
                    "kind": "rate_limit",
                    "at": time.time(),
                    "provider": provider,
                    "model": model,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            should_alert = count == 1 or count % 5 == 0
        if should_alert:
            self._alert(
                f"RATE LIMIT #{count} on {provider}/{model}: {exc}",
                speak_text="Rate limit hit. API requests are being throttled.",
            )

    def record_slow(self, provider: str, model: str, elapsed_seconds: float) -> None:
        with self._lock:
            self.slow_requests += 1
            count = self.slow_requests
            self.events.append(
                {
                    "kind": "slow_request",
                    "at": time.time(),
                    "provider": provider,
                    "model": model,
                    "elapsed_seconds": round(elapsed_seconds, 1),
                }
            )
            should_alert = count == 1 or count % 5 == 0
        if should_alert:
            self._alert(
                f"SLOW REQUEST #{count} on {provider}/{model}: "
                f"{elapsed_seconds:.0f}s (possible rate-limit backoff)",
                speak_text="Slow API response. Possible rate limit backoff.",
            )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "rate_limit_hits": self.rate_limit_hits,
                "slow_requests": self.slow_requests,
                "events": list(self.events),
            }

    def progress_suffix(self) -> str:
        with self._lock:
            hits = self.rate_limit_hits
            slow = self.slow_requests
        if hits == 0 and slow == 0:
            return ""
        parts: list[str] = []
        if hits:
            parts.append(f"rate limits: {hits}")
        if slow:
            parts.append(f"slow requests: {slow}")
        return f" ({', '.join(parts)})"

    def print_summary(self) -> None:
        snap = self.snapshot()
        hits = snap["rate_limit_hits"]
        slow = snap["slow_requests"]
        if hits == 0 and slow == 0:
            return
        print("", flush=True)
        if hits:
            print(
                f"WARNING: {hits} rate-limit error(s) during this run. "
                "High concurrency may be throttled by the API.",
                flush=True,
            )
        if slow:
            print(
                f"WARNING: {slow} request(s) took >{SLOW_REQUEST_SECONDS:.0f}s "
                "(SDK may be retrying rate limits silently).",
                flush=True,
            )

    def _alert(self, message: str, speak_text: str) -> None:
        banner = f"\n*** {message} ***\n"
        print(banner, file=sys.stderr, flush=True)
        now = time.monotonic()
        with self._lock:
            if now - self._last_alert_at < 3.0:
                return
            self._last_alert_at = now
        self._beep()
        self._speak(speak_text)

    def _beep(self) -> None:
        if sys.platform != "win32":
            return
        import winsound

        winsound.MessageBeep(winsound.MB_ICONHAND)

    def _speak(self, text: str) -> None:
        if sys.platform != "win32":
            return
        subprocess.Popen(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Add-Type -AssemblyName System.Speech; "
                f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak({text!r})",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


monitor = RateLimitMonitor()
