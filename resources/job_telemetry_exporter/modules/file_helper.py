"""
File Helper Module

This module provides utility functions for file operations, including safe file removal.

Typical usage:
    from file_helper import safe_remove_file

    # Safely remove a file
    success = safe_remove_file("path/to/file.txt")
    if success:
        print("File removed successfully.")
    else:
        print("Failed to remove file.")
"""

import logging
import os

logger = logging.getLogger(__name__)


def safe_remove_file(filepath: str) -> bool:
    """
    Safely remove a file with proper error handling.

    Args:
        filepath: Path to the file to remove.

    Returns:
        True if file was removed successfully, False otherwise.
    """
    if not filepath or not os.path.exists(filepath):
        return True

    try:
        os.remove(filepath)
        logger.debug(f"Deleted temporary file: {filepath}")
        return True
    except OSError as e:
        logger.error(f"Error removing temporary file {filepath}: {e}")
        return False
