import os
import pandas as pd
import chromadb
import json
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Dict, Optional, Any
from config.config import settings
from utils.logger import logger
import time

class RAGEngineLocal:
    """
    RAG Engine using local embeddings (sentence-transformers) and ChromaDB.
    No external API calls are made during indexing or retrieval.
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(settings.RAG_DB_PATH)
        self._client = None
        self._collection = None
        self._embedding_function = None

        try:
            self._client = chromadb.PersistentClient(path=self.db_path)
            
            # Use a high-performance local embedding function
            # Force CPU if requested or use detected device
            device = "cpu" if settings.FORCE_CPU_EMBEDDINGS else settings.get_device()
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.RAG_EMBEDDING_MODEL,
                device=device
            )
            
            self._collection = self._client.get_or_create_collection(
                name=settings.RAG_COLLECTION_NAME,
                embedding_function=self._embedding_function
            )
            logger.info(f"Local RAG Engine initialized with {self._collection.count()} documents using {settings.RAG_EMBEDDING_MODEL}.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}", exc_info=True)
            self._client = None
            self._collection = None

    def index_data(self, force: bool = False):
        """Load and index transaction, complaint, and expert QA data."""
        if not self._collection:
            logger.error("Attempted to index data but ChromaDB collection is not initialized.")
            return

        if self._collection.count() > 0 and not force:
            logger.info("ChromaDB already has documents. Skipping indexing (use force=True to re-index).")
            return

        logger.info("Indexing local knowledge base...")
        start_time = time.time()
        
        # 1. Index CFPB Complaints (the contextual knowledge base)
        cfpb_path = settings.CFPB_DATA_PATH
        if cfpb_path.exists():
            logger.info(f"Indexing CFPB complaints from {cfpb_path}...")
            try:
                # Load a larger chunk for better coverage
                df_cfpb = pd.read_csv(cfpb_path, nrows=1000) if pd is not None else None
                if df_cfpb is not None:
                    documents, metadatas, ids = [], [], []
                    for i, row in df_cfpb.iterrows():
                        issue = row.get('Issue', 'Financial Issue')
                        complaint = row.get('Consumer complaint narrative', '')
                        if pd.isna(complaint) or complaint == '':
                            complaint = f"Consumer reported an issue regarding {issue} with {row.get('Company', 'a financial institution')}."
                        
                        text = f"CFPB Complaint: {issue}. Narrative: {complaint}"
                        documents.append(text)
                        metadatas.append({
                            "type": "complaint", 
                            "company": str(row.get("Company", "Unknown")), 
                            "state": str(row.get("State", "Unknown"))
                        })
                        ids.append(f"cfpb_{i}")
                    
                    # Batch add
                    batch_size = 100
                    for j in range(0, len(documents), batch_size):
                        self._collection.add(
                            documents=documents[j:j+batch_size],
                            metadatas=metadatas[j:j+batch_size],
                            ids=ids[j:j+batch_size]
                        )
                else:
                    logger.warning("Pandas not available. Skipping CFPB indexing.")
            except Exception as e:
                logger.error(f"❌ Error indexing CFPB data: {e}", exc_info=True)
        else:
            logger.warning(f"CFPB data not found at {cfpb_path}. Skipping.")

        duration = time.time() - start_time
        logger.info(f"✅ Indexed {self._collection.count()} total items in {duration:.2f}s")

        # 2. Index Fraud Expert Q&A (the expert intelligence base)
        qa_path = settings.DATA_DIR / "fraud_detection_qa_dataset.json"
        if qa_path.exists():
            logger.info(f"Indexing expert fraud intelligence from {qa_path}...")
            with open(qa_path, 'r') as f:
                qa_data = json.load(f)
            
            documents, metadatas, ids = [], [], []
            for qa in qa_data.get("qa_pairs", []):
                category = qa.get("category", "General")
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                
                text = f"Expert Intelligence ({category}): {question} - {answer}"
                documents.append(text)
                metadatas.append({
                    "type": "expert_qa", 
                    "category": category, 
                    "difficulty": qa.get("difficulty", "N/A")
                })
                ids.append(f"qa_{qa.get('id', 'unknown')}")
            
            if documents:
                self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
            
        indexing_duration = time.time() - start_time
        logger.info(f"Summary: Indexed {self._collection.count()} items in {indexing_duration:.2f}s.")

    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Semantic search with re-ranking."""
        if not self._collection:
            return []

        n_results = n_results * 2
        
        start_time = time.time()
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        query_duration = time.time() - start_time
        logger.debug(f"RAG query took {query_duration:.4f}s for '{query_text[:30]}...'")

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        parsed = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            conf = max(0, 1 - (dist / settings.RAG_DISTANCE_SCALE))
            parsed.append({
                "text": str(doc),
                "metadata": meta,
                "confidence": conf,
                "type": meta.get("type", "unknown")
            })

        expert_qa = [r for r in parsed if r["type"] == "expert_qa" and r["confidence"] > settings.RAG_CONFIDENCE_THRESHOLD]
        other_data = [r for r in parsed if r not in expert_qa]

        expert_qa.sort(key=lambda x: x["confidence"], reverse=True)
        other_data.sort(key=lambda x: x["confidence"], reverse=True)

        merged = expert_qa + other_data
        return merged[:n_results]

    def get_context_for_query(self, query_text: str) -> str:
        """Returns a formatted context string for the LLM."""
        results = self.query(query_text, n_results=settings.RAG_CONTEXT_COUNT)
        if not results:
            return "No relevant context found."
        
        context = []
        for i, res in enumerate(results, 1):
            source = str(res["type"]).upper()
            context.append(f"[{source} {i}]: {res['text']}")
        
        return "\n\n".join(context)

if __name__ == "__main__":
    engine = RAGEngineLocal()
    engine.index_data(force=True)
