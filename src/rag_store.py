import os
import logging
import uuid
from typing import Dict, Any, List
from datetime import datetime

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "sase-catalysts"

pc = None
index = None
embeddings = None

def init_pinecone():
    global pc, index, embeddings
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        logger.warning("Pinecone or OpenAI API key missing. RAG Store will be disabled.")
        return False
        
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        
        # Check if index exists, create if not
        if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
            logger.info(f"Creating Pinecone index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536, # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        index = pc.Index(INDEX_NAME)
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        return False

def query_catalyst_history(ticker: str, catalyst_enum: str, lookback_count: int = 5) -> Dict[str, float]:
    """
    Queries the vector store for similar historical catalysts.
    Aggregates metrics: avg_1h_return, avg_4h_return, win_rate.
    """
    if not pc and not init_pinecone():
        return {"avg_1h_return": 0.0, "avg_4h_return": 0.0, "win_rate": 0.5, "count": 0}
        
    try:
        # Embed the search query
        query_text = f"Catalyst type: {catalyst_enum}"
        query_vector = embeddings.embed_query(query_text)
        
        # Query Pinecone
        response = index.query(
            vector=query_vector,
            top_k=lookback_count,
            include_metadata=True
        )
        
        matches = response.get('matches', [])
        if not matches:
            return {"avg_1h_return": 0.0, "avg_4h_return": 0.0, "win_rate": 0.5, "count": 0}
            
        returns_1h = []
        returns_4h = []
        wins = 0
        
        for match in matches:
            meta = match['metadata']
            ret_1h = meta.get('return_1h', 0.0)
            ret_4h = meta.get('return_4h', 0.0)
            
            returns_1h.append(ret_1h)
            returns_4h.append(ret_4h)
            if ret_1h > 0 or ret_4h > 0: # Simple win definition
                wins += 1
                
        count = len(matches)
        return {
            "avg_1h_return": sum(returns_1h) / count if count else 0.0,
            "avg_4h_return": sum(returns_4h) / count if count else 0.0,
            "win_rate": wins / count if count else 0.5,
            "count": count
        }
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return {"avg_1h_return": 0.0, "avg_4h_return": 0.0, "win_rate": 0.5, "count": 0}

def store_catalyst_event(ticker: str, catalyst_enum: str, headline: str, 
                         return_1h: float, return_4h: float, timestamp: datetime = None):
    """
    Saves a resolved event to Pinecone post-market for future RAG queries.
    """
    if not pc and not init_pinecone():
        return False
        
    try:
        if not timestamp:
            timestamp = datetime.now()
            
        text = f"Ticker {ticker} experienced {catalyst_enum}. Headline: {headline}"
        vector = embeddings.embed_query(text)
        
        vector_id = str(uuid.uuid4())
        
        metadata = {
            "ticker": ticker,
            "catalyst_enum": catalyst_enum,
            "headline": headline,
            "return_1h": return_1h,
            "return_4h": return_4h,
            "timestamp": timestamp.isoformat()
        }
        
        index.upsert(vectors=[{"id": vector_id, "values": vector, "metadata": metadata}])
        logger.info(f"Stored catalyst event for {ticker} -> {vector_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to store catalyst event: {e}")
        return False
