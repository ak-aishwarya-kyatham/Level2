import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class ChunkingAgent:
    """
    Chunks text using LangChain RecursiveCharacterTextSplitter.
    Configured for chunk_size=1000 and chunk_overlap=200.
    """
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    async def run(self, text: str) -> List[str]:
        logger.info("Chunking document content...")
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Generated {len(chunks)} chunks.")
        return chunks
