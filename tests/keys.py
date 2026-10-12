"""A private key block to put in a file, built when the test runs.

The body is the bytes 0 to 255, base64: no scanner takes it for a key, and no credential-shaped
literal is in this repository, which is public.
"""

from __future__ import annotations

import base64


def pem(kind: str = "RSA PRIVATE KEY", lines: int = 4) -> tuple[list[str], list[str]]:
    """The lines of a block, BEGIN and END included, and just the lines of its body."""
    body = base64.b64encode(bytes(range(256)) * 2).decode()
    rows = [body[i : i + 64] for i in range(0, 64 * lines, 64)]
    return [f"-----BEGIN {kind}-----", *rows, f"-----END {kind}-----"], rows
