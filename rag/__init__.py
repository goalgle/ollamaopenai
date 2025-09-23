"""
RAG (Retrieval-Augmented Generation) Module
Core components for vector database operations and knowledge management
"""

__version__ = "0.1.0"
__author__ = "Ollama Agents Team"

# Core imports
from .knowledge_manager import KnowledgeManager

__all__ = [
    "KnowledgeManager",
]