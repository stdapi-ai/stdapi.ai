"""Rendering token counts the way vendors quote them."""

from __future__ import annotations


def parse_tokens(value: object) -> int | None:
    """Read a token count that may be written for a human.

    Args:
        value: An integer, or a string such as ``163,840``, ``200K``, ``1M`` or
            ``128K tokens`` — the AWS model cards always name the unit.

    Returns:
        The count, or ``None`` when the value cannot be read.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value or None
    if not isinstance(value, str):
        return None
    text = value.strip().lower().removesuffix("tokens").removesuffix("token")
    text = text.strip().replace(",", "").replace(" ", "")
    multiplier = 1
    if text[-1:].upper() == "K":
        multiplier, text = 1000, text[:-1]
    elif text[-1:].upper() == "M":
        multiplier, text = 1_000_000, text[:-1]
    try:
        return int(float(text) * multiplier) or None
    except ValueError:
        return None


def format_tokens(value: object) -> str | None:
    """Render a token count the way a reader reads it.

    Vendors quote both bases: 128000 is written 128K, and 131072 is also
    written 128K. Decimal is tried first and binary only when it does not
    divide, which reproduces what the vendor wrote in either case.

    Args:
        value: A token count in any form :func:`parse_tokens` accepts.

    Returns:
        ``1M``, ``200K``, the plain number, or ``None``.
    """
    count = parse_tokens(value)
    if count is None or count <= 0:
        return None
    # A quotient of 1000 or more belongs in the next unit up: 1050000 is 1.05M,
    # which a reader compares, not 1050K, which nobody writes.
    for unit, decimal, binary in (("M", 1_000_000, 1024 * 1024), ("K", 1000, 1024)):
        if count % decimal == 0 and decimal <= count < decimal * 1000:
            return f"{count // decimal}{unit}"
        if count % binary == 0 and binary <= count < binary * 1000:
            return f"{count // binary}{unit}"
    # Neither base divides it exactly: keep the unit a reader compares in rather
    # than print 1047576, which no vendor writes. Decimals earn their place only
    # while the whole part is small — 1.05M is a number, 262.14K is noise.
    for unit, size in (("M", 1_000_000), ("K", 1000)):
        if count >= size:
            scaled = count / size
            if scaled >= 100:
                return f"{round(scaled)}{unit}"
            return f"{scaled:.2f}".rstrip("0").rstrip(".") + unit
    return str(count)
