def format_context(chunks):
    return "\n\n".join(chunks)


def validate_query(query: str):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
