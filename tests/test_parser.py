from app.parser import compute_error_hash, normalize_raw_text

PYTHON_TRACE = """Traceback (most recent call last):
  File "/app/checkout/orders.py", line 88, in create_order
    session.execute(stmt)
  File "/app/db/pool.py", line 12, in acquire
    raise TimeoutError("pool exhausted")
TimeoutError: QueuePool limit of size 5 overflow 10 reached
"""


def java_trace(frames: int) -> str:
    lines = ["java.lang.NullPointerException: Cannot invoke \"User.getId()\" because \"user\" is null"]
    lines += [f"\tat com.shop.Service.method{i}(Service.java:{i + 10})" for i in range(frames)]
    return "\n".join(lines)


def test_empty_and_blank_input_normalize_to_empty_string():
    assert normalize_raw_text("") == ""
    assert normalize_raw_text("\n  \n\t\n") == ""


def test_message_lines_come_before_frames():
    # The message line arrives last in a Python traceback, but carries the
    # signal, so it is moved in front of the frames.
    lines = normalize_raw_text(PYTHON_TRACE).splitlines()
    assert lines[0] == "Traceback (most recent call last):"
    assert lines[1] == "TimeoutError: QueuePool limit of size 5 overflow 10 reached"
    assert lines[2:] == [
        'File "/app/checkout/orders.py", line 88, in create_order',
        'File "/app/db/pool.py", line 12, in acquire',
    ]


def test_python_source_snippet_under_a_frame_is_dropped():
    normalized = normalize_raw_text(PYTHON_TRACE)
    assert "session.execute(stmt)" not in normalized
    assert 'raise TimeoutError("pool exhausted")' not in normalized


def test_frames_beyond_the_limit_are_truncated_with_a_note():
    lines = normalize_raw_text(java_trace(frames=8), max_frames=5).splitlines()
    assert len(lines) == 1 + 5 + 1  # message, kept frames, note
    assert lines[1] == "at com.shop.Service.method0(Service.java:10)"
    assert lines[-1] == "... (3 more frames truncated)"


def test_no_note_when_frames_fit():
    normalized = normalize_raw_text(java_trace(frames=5), max_frames=5)
    assert "truncated" not in normalized


def test_go_goroutine_and_file_lines_count_as_frames():
    raw = "panic: runtime error: invalid memory address\n\ngoroutine 1 [running]:\nmain.main()\n\t/app/main.go:20 +0x1b"
    lines = normalize_raw_text(raw, max_frames=1).splitlines()
    # "main.main()" is not a frame pattern, so it stays a message line.
    assert lines == ["panic: runtime error: invalid memory address", "main.main()", "goroutine 1 [running]:",
                     "... (1 more frames truncated)"]


def test_cosmetic_differences_produce_the_same_hash():
    a = normalize_raw_text(java_trace(frames=3))
    b = normalize_raw_text("\r\n\r\n" + java_trace(frames=3).replace("\n", "\r\n\r\n") + "   \r\n")
    assert a == b
    assert compute_error_hash(a) == compute_error_hash(b)


def test_hash_is_hex_sha256_and_tells_errors_apart():
    h = compute_error_hash("TimeoutError: pool exhausted")
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    assert h != compute_error_hash("TimeoutError: pool exhausted!")
