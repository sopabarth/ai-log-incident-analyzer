"""
Raw log normalization.

Runs before anything is sent to the LLM. Two jobs:
  1. Trim stack traces down to the part that actually carries signal
     (the error message + the top N frames), so we don't burn tokens/cost
     on 300-line traces where the last 280 lines are framework internals.
  2. Produce a stable, whitespace/order-independent string to hash for
     dedup (see app/dedup.py) - two occurrences of "the same" error should
     normalize to the same text even if timestamps or line spacing differ.
"""

import hashlib
import re

# Lines that look like stack frames across the languages we expect to see:
# Python ("File \"x.py\", line 12"), Java/Kotlin/JS ("\tat pkg.Class.method"),
# Go ("\t/path/file.go:20 +0x1b" or "goroutine 1 [running]:").
_FRAME_LINE_RE = re.compile(
    r'^\s*(at\s|File "|goroutine\s|\t|\S+\.(java|go|py|js|kt):\d+)'
)

DEFAULT_MAX_FRAMES = 5


def normalize_raw_text(raw_text: str, max_frames: int = DEFAULT_MAX_FRAMES) -> str:
    """
    Collapse a raw log/trace blob into: all non-frame lines (error message,
    context lines) followed by up to `max_frames` stack frame lines.

    Message lines always survive truncation - they carry the actual
    "what happened" signal. Frames beyond the limit are dropped with a
    note, since the first few frames near the failure point are almost
    always enough to diagnose root cause; deep framework/stdlib frames add
    tokens without adding signal.

    One special case: in a Python traceback, the source-code line echoed
    under a `File "...", line N` frame (e.g. `conn = pool.acquire(...)`)
    is redundant with the frame reference itself and isn't a message - it's
    dropped rather than kept, so it doesn't get misclassified as signal.
    """
    raw_lines = [line for line in raw_text.strip("\n").splitlines() if line.strip()]
    if not raw_lines:
        return ""

    message_lines: list[str] = []
    frame_lines: list[str] = []
    prev_was_py_file_frame = False
    for line in raw_lines:
        stripped = line.strip()
        if _FRAME_LINE_RE.match(stripped):
            frame_lines.append(stripped)
            prev_was_py_file_frame = stripped.startswith('File "')
            continue
        if prev_was_py_file_frame and line[:1].isspace():
            # source snippet echoed under a Python "File ..." frame - noise, drop it
            prev_was_py_file_frame = False
            continue
        prev_was_py_file_frame = False
        message_lines.append(stripped)

    kept_frames = frame_lines[:max_frames]
    result = message_lines + kept_frames
    if len(frame_lines) > max_frames:
        result.append(f"... ({len(frame_lines) - max_frames} more frames truncated)")

    return "\n".join(result)


def compute_error_hash(normalized_text: str) -> str:
    """
    Hash of the *normalized* text (not the raw input) so that cosmetic
    differences - timestamps, extra whitespace, ordering - don't produce
    different hashes for what is semantically the same recurring error.
    Used by app/dedup.py to avoid re-analyzing the same error hundreds of
    times in a burst.
    """
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
