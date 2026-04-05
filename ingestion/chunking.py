def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list:
    """
    Split text into overlapping chunks with validation
    """

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []

    start = 0
    total_words = len(words)

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk = words[start:end]

        chunks.append(" ".join(chunk))

        start += chunk_size - overlap  # safer increment

    return chunks
