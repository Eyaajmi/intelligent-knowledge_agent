def split_text(text, chunk_size=300, overlap=50):

    text = text.strip()

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:

            if current_chunk:
                current_chunk += "\n\n"

            current_chunk += paragraph

        else:

           
            if current_chunk:
                chunks.append(current_chunk)

           
            overlap_text = current_chunk[-overlap:]

            current_chunk = overlap_text + "\n\n" + paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks