import os
import re
from glob import glob

import nltk

class DocumentProcessing:
    """
    A class used for processing documents including reading, trimming whitespace,
    and splitting documents into sentence chunks.

    Methods
    -------
    read_text_file(file_path: str) -> str
        Reads the content of a text file.
    
    trim_white_space(text: str) -> str
        Trims extra whitespace from the given text.
    
    split_document(document_filename: str, sentences_per_chunk: int) -> list
        Splits the document into chunks of specified number of sentences.
    """

    def __init__(self):
        """Initializes the DocumentProcessing class."""
        pass

    def read_text_file(self, file_path):
        """
        Reads the content of a text file.

        :param file_path: The path to the text file.
        :type file_path: str
        :return: The content of the text file or an error message.
        :rtype: str
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            return f"The file at {file_path} was not found."
        except Exception as e:
            return f"An error occurred: {e}"

    def trim_white_space(self, text):
        """
        Trims extra whitespace from the given text.

        :param text: The text to be trimmed.
        :type text: str
        :return: The trimmed text.
        :rtype: str
        """
        return ' '.join(text.split())
    
    def sentence_chunking(self, document_filename, overlap_size):
        """
        Splits the document into chunks of specified number of sentences.

        :param document_filename: The filename of the document to be split.
        :type document_filename: str
        :param sentences_per_chunk: The number of sentences per chunk.
        :type sentences_per_chunk: int
        :return: A list of chunks, each containing the specified number of sentences.
        :rtype: list
        """
        text = self.read_text_file(document_filename)
        
        # Preprocessing
        text = self.trim_white_space(text)

        # Split documents into sentence chunks
        sentences = nltk.sent_tokenize(text)
    
        # Create chunks of the specified number of sentences
        chunks = [' '.join(sentences[i:i + overlap_size]) for i in range(0, len(sentences), overlap_size)]
        return chunks
    
    def fixed_length_chunking(self, document_filename, fixed_length, overlap_size=2):
        text = self.read_text_file(document_filename)
        
        # Preprocessing
        text = self.trim_white_space(text)

        # Split documents into word chunks
        words = nltk.word_tokenize(text)

        if not words:
            return []

        step = fixed_length - overlap_size
        if step <= 0:
            step = 1  # Ensure the step size is positive
            
        print(f"Total words: {len(words)}")
        print(f"Fixed length: {fixed_length}")
        print(f"Overlap size: {overlap_size}")
        print(f"Step size: {step}")

        chunks = [' '.join(words[i:i + fixed_length]) for i in range(0, len(words), step)]
        return chunks