"""
Medical Knowledge Retriever (RAG Module)
==========================================
FAISS-based retrieval-augmented generation for injecting
relevant medical knowledge into VQA prompts.

Uses sentence-transformers for semantic embedding and FAISS
for efficient nearest-neighbor search.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from medical_vqa.knowledge.knowledge_base import get_medical_knowledge_base


class MedicalKnowledgeRetriever:
    """
    RAG-based medical knowledge retriever.
    
    Pipeline:
        1. Encode all medical knowledge entries using sentence-transformers
        2. Build FAISS index for fast similarity search
        3. At inference: encode query → search index → return top-k facts
        4. Format retrieved facts for prompt injection
    
    Attributes:
        embedding_model_name: Name of the sentence-transformer model.
        index: FAISS index for vector similarity search.
        knowledge_entries: List of knowledge dictionaries.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_dir: Optional[str] = None,
        top_k: int = 3
    ):
        """
        Args:
            embedding_model_name: HuggingFace model for text embeddings.
            index_dir: Directory to save/load the FAISS index.
            top_k: Default number of results to retrieve.
        """
        self.embedding_model_name = embedding_model_name
        self.index_dir = Path(index_dir) if index_dir else None
        self.top_k = top_k
        
        self.model = None
        self.index = None
        self.knowledge_entries: List[Dict[str, str]] = []
        self.embeddings: Optional[np.ndarray] = None
        
        logger.info(f"MedicalKnowledgeRetriever initialized with {embedding_model_name}")
    
    def _load_embedding_model(self):
        """Lazy-load the sentence-transformer embedding model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
    
    def build_index(self, additional_knowledge: Optional[List[Dict[str, str]]] = None):
        """
        Build the FAISS index from the medical knowledge base.
        
        Steps:
            1. Load knowledge entries
            2. Create text representations for embedding
            3. Encode with sentence-transformer
            4. Build FAISS index (Inner Product for cosine similarity)
        
        Args:
            additional_knowledge: Optional extra knowledge entries to add.
        """
        import faiss
        
        self._load_embedding_model()
        
        # Load knowledge base
        self.knowledge_entries = get_medical_knowledge_base()
        if additional_knowledge:
            self.knowledge_entries.extend(additional_knowledge)
        
        logger.info(f"Building index with {len(self.knowledge_entries)} knowledge entries...")
        
        # Create text representations (concept + definition for richer embedding)
        texts = [
            f"{entry['concept']}: {entry['definition']}"
            for entry in self.knowledge_entries
        ]
        
        # Encode all texts
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,  # For cosine similarity via inner product
            batch_size=32
        )
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product ≈ cosine sim (normalized)
        self.index.add(self.embeddings.astype(np.float32))
        
        logger.info(
            f"FAISS index built: {self.index.ntotal} vectors, "
            f"dimension={dimension}"
        )
        
        # Save index if directory specified
        if self.index_dir:
            self.save_index()
    
    def save_index(self):
        """Save the FAISS index and knowledge entries to disk."""
        import faiss
        
        if self.index is None:
            logger.warning("No index to save. Build index first.")
            return
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = self.index_dir / "medical_knowledge.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save knowledge entries
        entries_path = self.index_dir / "knowledge_entries.pkl"
        with open(entries_path, "wb") as f:
            pickle.dump(self.knowledge_entries, f)
        
        logger.info(f"Index saved to {self.index_dir}")
    
    def load_index(self):
        """Load a previously saved FAISS index."""
        import faiss
        
        if self.index_dir is None:
            raise ValueError("index_dir not specified")
        
        index_path = self.index_dir / "medical_knowledge.index"
        entries_path = self.index_dir / "knowledge_entries.pkl"
        
        if not index_path.exists() or not entries_path.exists():
            logger.warning("Saved index not found. Building new index...")
            self.build_index()
            return
        
        self.index = faiss.read_index(str(index_path))
        
        with open(entries_path, "rb") as f:
            self.knowledge_entries = pickle.load(f)
        
        self._load_embedding_model()
        
        logger.info(
            f"Loaded index: {self.index.ntotal} vectors, "
            f"{len(self.knowledge_entries)} entries"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Retrieve the most relevant medical knowledge for a query.
        
        Args:
            query: Search query (typically the VQA question).
            top_k: Number of results to return (overrides default).
        
        Returns:
            List of dicts with keys: concept, definition, category, source, score.
        """
        if self.index is None:
            logger.warning("Index not built. Building now...")
            self.build_index()
        
        k = top_k or self.top_k
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype(np.float32)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.knowledge_entries):
                entry = self.knowledge_entries[idx].copy()
                entry["score"] = float(score)
                results.append(entry)
        
        logger.debug(
            f"Retrieved {len(results)} facts for query: '{query[:50]}...' "
            f"(top score: {results[0]['score']:.3f})" if results else ""
        )
        
        return results
    
    def format_knowledge(
        self,
        facts: List[Dict[str, any]],
        max_tokens: int = 200
    ) -> str:
        """
        Format retrieved knowledge facts into a string for prompt injection.
        
        Args:
            facts: List of knowledge entries from retrieve().
            max_tokens: Approximate max character count (rough token estimate).
        
        Returns:
            Formatted knowledge string for the model prompt.
        """
        if not facts:
            return "No specific medical knowledge retrieved."
        
        formatted_parts = []
        total_chars = 0
        
        for i, fact in enumerate(facts, 1):
            entry_text = f"{i}. {fact['concept']}: {fact['definition']}"
            
            # Rough check against token limit (1 token ≈ 4 chars)
            if total_chars + len(entry_text) > max_tokens * 4:
                # Truncate to fit
                remaining = max_tokens * 4 - total_chars
                if remaining > 50:
                    entry_text = entry_text[:remaining] + "..."
                    formatted_parts.append(entry_text)
                break
            
            formatted_parts.append(entry_text)
            total_chars += len(entry_text)
        
        return "\n".join(formatted_parts)
    
    def retrieve_and_format(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_tokens: int = 200
    ) -> str:
        """
        Convenience method: retrieve facts and format for prompt.
        
        Args:
            query: Search query.
            top_k: Number of results.
            max_tokens: Maximum tokens for formatted output.
        
        Returns:
            Formatted knowledge string.
        """
        facts = self.retrieve(query, top_k)
        return self.format_knowledge(facts, max_tokens)
