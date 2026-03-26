"""Document processing utilities for extracting and chunking text from files."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported file extensions and their categories
PLAIN_TEXT_EXTENSIONS = {'.txt', '.md', '.rst', '.csv', '.log', '.yaml', '.yml', '.json', '.toml', '.ini', '.env'}
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs',
    '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.sh', '.bash',
    '.sql', '.html', '.css', '.xml', '.r', '.m', '.lua',
}
PDF_EXTENSIONS = {'.pdf'}
DOCX_EXTENSIONS = {'.docx', '.doc'}

ALL_SUPPORTED_EXTENSIONS = (
    PLAIN_TEXT_EXTENSIONS | CODE_EXTENSIONS | PDF_EXTENSIONS | DOCX_EXTENSIONS
)


@dataclass
class DocumentChunk:
    """A chunk of text extracted from a document."""

    content: str
    chunk_index: int
    total_chunks: int
    source_path: str
    start_char: int
    end_char: int
    metadata: dict


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    chunk_size: int = 1500        # Target characters per chunk
    chunk_overlap: int = 200      # Characters of overlap between chunks
    min_chunk_size: int = 100     # Minimum chunk size (discard smaller)
    respect_paragraphs: bool = True  # Try to split on paragraph boundaries


def extract_text(file_path: str) -> tuple[str, dict]:
    """
    Extract text content from a file.

    Args:
        file_path: Absolute or relative path to the file

    Returns:
        Tuple of (extracted_text, metadata_dict)

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file type is not supported
        RuntimeError: If extraction fails
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f'File not found: {file_path}')

    if not path.is_file():
        raise ValueError(f'Path is not a file: {file_path}')

    ext = path.suffix.lower()
    file_size = path.stat().st_size

    metadata = {
        'file_name': path.name,
        'file_path': str(path),
        'file_extension': ext,
        'file_size_bytes': file_size,
    }

    if ext in PLAIN_TEXT_EXTENSIONS | CODE_EXTENSIONS:
        text = _extract_plain_text(path)
    elif ext in PDF_EXTENSIONS:
        text = _extract_pdf(path)
    elif ext in DOCX_EXTENSIONS:
        text = _extract_docx(path)
    else:
        raise ValueError(
            f'Unsupported file type: {ext}. '
            f'Supported types: {", ".join(sorted(ALL_SUPPORTED_EXTENSIONS))}'
        )

    metadata['char_count'] = len(text)
    metadata['word_count'] = len(text.split())
    metadata['line_count'] = text.count('\n') + 1

    return text, metadata


def _extract_plain_text(path: Path) -> str:
    """Extract text from plain text / code files."""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f'Could not decode file {path} with any known encoding')


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF files using pypdf."""
    try:
        import pypdf  # type: ignore
    except ImportError:
        raise RuntimeError(
            'pypdf is required for PDF extraction. Install with: pip install pypdf'
        )

    text_parts = []
    with open(path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ''
            if page_text.strip():
                text_parts.append(f'[Page {page_num + 1}]\n{page_text}')

    return '\n\n'.join(text_parts)


def _extract_docx(path: Path) -> str:
    """Extract text from Word documents using python-docx."""
    try:
        import docx  # type: ignore
    except ImportError:
        raise RuntimeError(
            'python-docx is required for Word document extraction. '
            'Install with: pip install python-docx'
        )

    doc = docx.Document(str(path))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return '\n\n'.join(paragraphs)


def chunk_text(
    text: str,
    config: ChunkingConfig | None = None,
) -> list[tuple[str, int, int]]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to chunk
        config: Chunking configuration (uses defaults if None)

    Returns:
        List of (chunk_text, start_char, end_char) tuples
    """
    if config is None:
        config = ChunkingConfig()

    if not text.strip():
        return []

    chunks: list[tuple[str, int, int]] = []

    if config.respect_paragraphs:
        chunks = _chunk_by_paragraphs(text, config)
    else:
        chunks = _chunk_by_characters(text, config)

    # Filter out chunks that are too small
    chunks = [(c, s, e) for c, s, e in chunks if len(c.strip()) >= config.min_chunk_size]

    return chunks


def _chunk_by_paragraphs(
    text: str, config: ChunkingConfig
) -> list[tuple[str, int, int]]:
    """
    Chunk text by trying to respect paragraph / sentence boundaries.
    Falls back to hard character splitting when paragraphs are too large.
    """
    # Split into paragraphs (double newline) first
    paragraph_pattern = re.compile(r'\n\s*\n')
    paragraphs: list[tuple[str, int]] = []  # (text, start_char)

    last_end = 0
    for match in paragraph_pattern.finditer(text):
        para_text = text[last_end:match.start()]
        if para_text.strip():
            paragraphs.append((para_text, last_end))
        last_end = match.end()

    # Remaining text after last double-newline
    remaining = text[last_end:]
    if remaining.strip():
        paragraphs.append((remaining, last_end))

    if not paragraphs:
        return _chunk_by_characters(text, config)

    chunks: list[tuple[str, int, int]] = []
    current_parts: list[tuple[str, int]] = []  # (text, start_char)
    current_len = 0

    for para_text, para_start in paragraphs:
        para_len = len(para_text)

        # If this single paragraph exceeds chunk_size, hard-split it
        if para_len > config.chunk_size:
            # Flush current accumulation first
            if current_parts:
                chunk_text_str = '\n\n'.join(p for p, _ in current_parts)
                chunk_start = current_parts[0][1]
                chunk_end = chunk_start + len(chunk_text_str)
                chunks.append((chunk_text_str, chunk_start, chunk_end))
                current_parts = []
                current_len = 0

            # Hard-split the oversized paragraph
            sub_chunks = _chunk_by_characters(
                para_text,
                config,
                offset=para_start,
            )
            chunks.extend(sub_chunks)
            continue

        # Would adding this paragraph exceed the chunk size?
        separator_len = 2 if current_parts else 0  # "\n\n"
        if current_len + separator_len + para_len > config.chunk_size and current_parts:
            # Save current chunk
            chunk_text_str = '\n\n'.join(p for p, _ in current_parts)
            chunk_start = current_parts[0][1]
            chunk_end = chunk_start + len(chunk_text_str)
            chunks.append((chunk_text_str, chunk_start, chunk_end))

            # Start new chunk with overlap: keep last paragraph(s) that fit in overlap window
            overlap_parts: list[tuple[str, int]] = []
            overlap_len = 0
            for prev_para, prev_start in reversed(current_parts):
                if overlap_len + len(prev_para) + 2 <= config.chunk_overlap:
                    overlap_parts.insert(0, (prev_para, prev_start))
                    overlap_len += len(prev_para) + 2
                else:
                    break

            current_parts = overlap_parts
            current_len = overlap_len

        current_parts.append((para_text, para_start))
        current_len += separator_len + para_len

    # Flush remaining
    if current_parts:
        chunk_text_str = '\n\n'.join(p for p, _ in current_parts)
        chunk_start = current_parts[0][1]
        chunk_end = chunk_start + len(chunk_text_str)
        chunks.append((chunk_text_str, chunk_start, chunk_end))

    return chunks


def _chunk_by_characters(
    text: str,
    config: ChunkingConfig,
    offset: int = 0,
) -> list[tuple[str, int, int]]:
    """Hard character-based chunking with overlap."""
    chunks: list[tuple[str, int, int]] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + config.chunk_size, text_len)

        # Try to break on a sentence boundary near the end
        if end < text_len:
            # Look back up to 200 chars for a sentence end
            search_region = text[max(end - 200, start):end]
            sentence_end = max(
                search_region.rfind('. '),
                search_region.rfind('.\n'),
                search_region.rfind('! '),
                search_region.rfind('? '),
            )
            if sentence_end != -1:
                end = max(end - 200, start) + sentence_end + 2  # include the punctuation + space

        chunk = text[start:end]
        if chunk.strip():
            chunks.append((chunk, offset + start, offset + end))

        # Advance with overlap
        start = end - config.chunk_overlap
        if start <= (end - config.chunk_size):
            # Prevent infinite loop if overlap >= chunk_size
            start = end

    return chunks


def create_document_chunks(
    file_path: str,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """
    Extract text from a file and split it into chunks.

    Args:
        file_path: Path to the document
        config: Optional chunking configuration

    Returns:
        List of DocumentChunk objects ready to be added to the graph
    """
    if config is None:
        config = ChunkingConfig()

    text, metadata = extract_text(file_path)
    raw_chunks = chunk_text(text, config)
    total = len(raw_chunks)

    document_chunks = []
    for idx, (chunk_content, start_char, end_char) in enumerate(raw_chunks):
        document_chunks.append(
            DocumentChunk(
                content=chunk_content,
                chunk_index=idx,
                total_chunks=total,
                source_path=metadata['file_path'],
                start_char=start_char,
                end_char=end_char,
                metadata=metadata,
            )
        )

    return document_chunks
