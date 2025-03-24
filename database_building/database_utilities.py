
import re

def split_into_sentences(text):
    """
    Splits Vietnamese text into sentences based on punctuation,
    retaining the punctuation marks.
    
    Parameters:
    - text (str): The input text to split.
    
    Returns:
    - List[str]: A list of sentences with punctuation.
    """
    # Regular expression to match sentences ending with ., !, or ?
    sentence_endings = re.compile(r'[^.!?]+[.!?]+')
    sentences = sentence_endings.findall(text)
    # Strip leading/trailing whitespace from each sentence
    sentences = [s.strip() for s in sentences]
    return sentences


def chunk_text_by_sentences_with_overlap(text, max_words=100, overlap_words=10):
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""
    current_word_count = 0
    overlap = []
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        
        if current_word_count + sentence_word_count > max_words:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Save the last 'overlap_words' from the current chunk
                overlap = current_chunk.strip().split()[-overlap_words:]
            # Start a new chunk with overlap
            current_chunk = " ".join(overlap) + " " + sentence + " "
            current_word_count = len(" ".join(overlap).split()) + sentence_word_count
        else:
            current_chunk += sentence + " "
            current_word_count += sentence_word_count
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def chunk_text_by_sentences(text, max_words=100):
    """
    Chunks text into segments where each chunk contains as many sentences as possible
    without exceeding the max_words limit.
    
    Parameters:
    - text (str): The text to chunk.
    - max_words (int): The maximum number of words per chunk.
    
    Returns:
    - List[str]: A list of text chunks.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""
    current_word_count = 0
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        
        # If adding the sentence exceeds the limit, finalize the current chunk
        if current_word_count + sentence_word_count > max_words:
            if current_chunk:  # Avoid adding empty chunks
                chunks.append(current_chunk.strip())
            # Start a new chunk with the current sentence
            current_chunk = sentence + " "
            current_word_count = sentence_word_count
        else:
            # Add the sentence to the current chunk
            current_chunk += sentence + " "
            current_word_count += sentence_word_count
    
    # Add the last chunk if it contains any sentences
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
