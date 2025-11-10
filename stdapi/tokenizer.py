"""Tokenizer."""

from stdapi.config import SETTINGS

if SETTINGS.tokens_estimation:
    from asyncio import to_thread

    from tiktoken import get_encoding

    def _estimate_token_count(
        *strings: str | None,
        encoding: str = SETTINGS.tokens_estimation_default_encoding,
    ) -> int:
        """Estimate the number of tokens in the given strings.

        Args:
            strings: The strings to tokenize.
            encoding: The encoding to use.

        Returns:
            The total number of tokens in all strings.
        """
        encode = get_encoding(encoding).encode
        count = 0
        for string in strings:
            if string:
                count += len(encode(string))
        return count

    async def estimate_token_count(
        *strings: str | None,
        encoding: str = SETTINGS.tokens_estimation_default_encoding,
    ) -> int | None:
        """Estimate the number of tokens in the given strings.

        Args:
            strings: The strings to tokenize.
            encoding: The encoding to use.

        Returns:
            The total number of tokens in all strings.
        """
        return await to_thread(_estimate_token_count, *strings, encoding=encoding)


else:
    # Estimation disabled

    async def estimate_token_count(
        *strings: str | None,  # noqa: ARG001
        encoding: str = "",  # noqa: ARG001
    ) -> int | None:
        """Estimate the number of tokens in the given string.

        Args:
            strings: The strings to tokenize.
            encoding: The encoding to use.

        Returns:
            The total number of tokens in all strings.
        """
        return None
