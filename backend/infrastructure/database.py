import os
from typing import List, Dict, Any, Optional

from opensearchpy import OpenSearch, RequestsHttpConnection

from backend.domain.interfaces import DatabaseInterface
from backend.domain.entities import Chunk, Document


class OpenSearchDatabase(DatabaseInterface):
    """Database implementation using OpenSearch"""
    
    def __init__(self, 
                 index_name: str = "documents",
                 hosts: List[Dict[str, Any]] = None,
                 http_auth: tuple = None,
                 use_ssl: bool = False,
                 verify_certs: bool = False,
                 ssl_show_warn: bool = False):
        """Initialize the OpenSearch database"""
        self.index_name = index_name
        
        # Set defaults if not provided
        if hosts is None:
            hosts = [{"host": os.environ.get("OPENSEARCH_HOST", "localhost"), 
                      "port": int(os.environ.get("OPENSEARCH_PORT", 9200))}]
        
        if http_auth is None:
            http_auth = (os.environ.get("OPENSEARCH_USER", "admin"), 
                         os.environ.get("OPENSEARCH_PASSWORD", "admin"))
        
        if os.environ.get("OPENSEARCH_USE_SSL"):
            use_ssl = os.environ.get("OPENSEARCH_USE_SSL", "false").lower() == "true"
        
        self.connection_params = {
            "hosts": hosts,
            "http_auth": http_auth,
            "use_ssl": use_ssl,
            "verify_certs": verify_certs,
            "ssl_show_warn": ssl_show_warn,
            "connection_class": RequestsHttpConnection
        }
        self.client = None
        
    def initialize(self) -> None:
        """Initialize the database, create indices if needed"""
        if self.client is None:
            self.client = OpenSearch(**self.connection_params)
        
        # Create index if it doesn't exist
        if not self.client.indices.exists(index=self.index_name):
            # Define mapping for the index
            mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "file_id": {"type": "keyword"},
                        "title": {"type": "text"},
                        "content": {"type": "text"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": 384  # Dimension of the all-MiniLM-L6-v2 model
                        }
                    }
                },
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                }
            }
            
            # Create the index with mappings
            self.client.indices.create(
                index=self.index_name,
                body=mapping
            )
            print(f"Created index: {self.index_name}")
    
    def index_document(self, document: Document) -> None:
        """Index a document and its chunks in the database"""
        self.initialize()
        
        # Index each chunk
        for chunk in document.chunks:
            doc = {
                "id": chunk.id,
                "file_id": chunk.file_id,
                "title": chunk.title,
                "content": chunk.content,
                "document_name": document.name
            }
            
            if chunk.embedding:
                doc["embedding"] = chunk.embedding
            
            self.client.index(
                index=self.index_name,
                body=doc,
                refresh=True
            )
    
    def search_chunks(self, query: str, search_type: str = "hybrid", limit: int = 5) -> List[Chunk]:
        """Search for chunks based on a query, using the specified search type"""
        self.initialize()
        
        if search_type == "semantic":
            return self._semantic_search(query, limit)
        elif search_type == "keyword":
            return self._keyword_search(query, limit)
        else:  # hybrid
            return self._hybrid_search(query, limit)
    
    def _semantic_search(self, query: str, limit: int = 5) -> List[Chunk]:
        """Perform semantic search using vector embeddings"""
        from backend.infrastructure.embedder import SentenceTransformerEmbedder
        
        # Create embedder just for this query if not passed in constructor
        embedder = SentenceTransformerEmbedder()
        query_embedding = embedder.embed_text(query)
        
        # Perform kNN search
        search_query = {
            "size": limit,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": limit
                    }
                }
            },
            "_source": ["id", "file_id", "title", "content", "document_name"]
        }
        
        response = self.client.search(
            body=search_query,
            index=self.index_name
        )
        
        # Process results
        chunks = []
        for hit in response["hits"]["hits"]:
            chunks.append(Chunk(
                id=hit["_source"].get("id", ""),
                file_id=hit["_source"]["file_id"],
                title=hit["_source"]["title"],
                content=hit["_source"]["content"]
            ))
        
        return chunks
    
    def _keyword_search(self, query: str, limit: int = 5) -> List[Chunk]:
        """Perform keyword-based search"""
        search_query = {
            "size": limit,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "content"],
                    "type": "best_fields"
                }
            },
            "_source": ["id", "file_id", "title", "content", "document_name"]
        }
        
        response = self.client.search(
            body=search_query,
            index=self.index_name
        )
        
        # Process results
        chunks = []
        for hit in response["hits"]["hits"]:
            chunks.append(Chunk(
                id=hit["_source"].get("id", ""),
                file_id=hit["_source"]["file_id"],
                title=hit["_source"]["title"],
                content=hit["_source"]["content"]
            ))
        
        return chunks
    
    def _hybrid_search(self, query: str, limit: int = 5) -> List[Chunk]:
        """Perform hybrid search using both keyword and semantic methods"""
        # Get results from both methods
        semantic_results = self._semantic_search(query, limit)
        keyword_results = self._keyword_search(query, limit)
        
        # Combine and deduplicate results
        combined_results = {}
        
        # Add semantic results with a score
        for i, chunk in enumerate(semantic_results):
            key = chunk.id
            score = (limit - i) / limit  # Normalize to 0-1
            combined_results[key] = {
                "chunk": chunk,
                "score": score
            }
        
        # Add keyword results with a score
        for i, chunk in enumerate(keyword_results):
            key = chunk.id
            score = (limit - i) / limit  # Normalize to 0-1
            
            if key in combined_results:
                # Combine scores
                combined_results[key]["score"] = (combined_results[key]["score"] + score) / 2
            else:
                combined_results[key] = {
                    "chunk": chunk,
                    "score": score
                }
        
        # Sort by score and take top results
        sorted_results = sorted(
            [v for k, v in combined_results.items()], 
            key=lambda x: x["score"], 
            reverse=True
        )
        
        # Get chunks from final results
        final_chunks = []
        for result in sorted_results[:limit]:
            final_chunks.append(result["chunk"])
        
        return final_chunks 