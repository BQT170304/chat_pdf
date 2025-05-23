import os
from typing import List, Dict, Any, Optional
import json
from opensearchpy import OpenSearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer

class OpenSearchClient:
    """Client for interacting with OpenSearch"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of OpenSearch client"""
        if cls._instance is None:
            cls._instance = OpenSearchClient(
                hosts=[{"host": os.environ.get("OPENSEARCH_HOST", "localhost"), 
                        "port": int(os.environ.get("OPENSEARCH_PORT", 9200))}],
                http_auth=(os.environ.get("OPENSEARCH_USER", "admin"), 
                           os.environ.get("OPENSEARCH_PASSWORD", "admin")),
                use_ssl=os.environ.get("OPENSEARCH_USE_SSL", "false").lower() == "true",
                verify_certs=False,
                ssl_show_warn=False
            )
        return cls._instance
    
    def __init__(self, hosts, http_auth, use_ssl=False, verify_certs=False, ssl_show_warn=False):
        """Initialize OpenSearch client"""
        self.client = OpenSearch(
            hosts=hosts,
            http_auth=http_auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_show_warn=ssl_show_warn,
            connection_class=RequestsHttpConnection
        )
        
        # Initialize the embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Create index if it doesn't exist
        self._create_index_if_not_exists("documents")
    
    def _create_index_if_not_exists(self, index_name: str):
        """Create index with mapping if it doesn't exist"""
        if not self.client.indices.exists(index=index_name):
            # Define mapping for the index
            mapping = {
                "mappings": {
                    "properties": {
                        "title": {"type": "text"},
                        "content": {"type": "text"},
                        "document_name": {"type": "keyword"},
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
                index=index_name,
                body=mapping
            )
            print(f"Created index: {index_name}")
    
    def index_document(self, index_name: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Index a document in OpenSearch
        
        Args:
            index_name: Name of the index
            document: Document to index
            
        Returns:
            Response from OpenSearch
        """
        response = self.client.index(
            index=index_name,
            body=document,
            refresh=True
        )
        return response
    
    def search(self, query: str, search_type: str = "hybrid", size: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents in OpenSearch
        
        Args:
            query: Query string
            search_type: Type of search (semantic, keyword, or hybrid)
            size: Number of results to return
            
        Returns:
            List of search results
        """
        if search_type == "semantic":
            return self._semantic_search(query, size)
        elif search_type == "keyword":
            return self._keyword_search(query, size)
        else:  # hybrid
            return self._hybrid_search(query, size)
    
    def _semantic_search(self, query: str, size: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector embeddings
        
        Args:
            query: Query string
            size: Number of results to return
            
        Returns:
            List of search results
        """
        # Generate embedding for the query
        query_embedding = self.model.encode(query).tolist()
        
        # Perform kNN search
        search_query = {
            "size": size,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": size
                    }
                }
            },
            "_source": ["title", "content", "document_name"]
        }
        
        response = self.client.search(
            body=search_query,
            index="documents"
        )
        
        # Process results
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "title": hit["_source"]["title"],
                "content": hit["_source"]["content"]
            })
        
        return results
    
    def _keyword_search(self, query: str, size: int = 5) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search
        
        Args:
            query: Query string
            size: Number of results to return
            
        Returns:
            List of search results
        """
        search_query = {
            "size": size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "content"],
                    "type": "best_fields"
                }
            },
            "_source": ["title", "content", "document_name"]
        }
        
        response = self.client.search(
            body=search_query,
            index="documents"
        )
        
        # Process results
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "title": hit["_source"]["title"],
                "content": hit["_source"]["content"]
            })
        
        return results
    
    def _hybrid_search(self, query: str, size: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using both keyword and semantic methods
        
        Args:
            query: Query string
            size: Number of results to return
            
        Returns:
            List of search results
        """
        # Get results from both methods
        semantic_results = self._semantic_search(query, size)
        keyword_results = self._keyword_search(query, size)
        
        # Combine and deduplicate results
        combined_results = {}
        
        # Add semantic results with a score
        for i, result in enumerate(semantic_results):
            key = f"{result['title']}_{result['content'][:50]}"
            score = (size - i) / size  # Normalize to 0-1
            combined_results[key] = {
                "title": result["title"],
                "content": result["content"],
                "score": score
            }
        
        # Add keyword results with a score
        for i, result in enumerate(keyword_results):
            key = f"{result['title']}_{result['content'][:50]}"
            score = (size - i) / size  # Normalize to 0-1
            
            if key in combined_results:
                # Combine scores
                combined_results[key]["score"] = (combined_results[key]["score"] + score) / 2
            else:
                combined_results[key] = {
                    "title": result["title"],
                    "content": result["content"],
                    "score": score
                }
        
        # Sort by score and take top results
        sorted_results = sorted(
            [v for k, v in combined_results.items()], 
            key=lambda x: x["score"], 
            reverse=True
        )
        
        # Remove score field from final results
        final_results = []
        for result in sorted_results[:size]:
            final_results.append({
                "title": result["title"],
                "content": result["content"]
            })
        
        return final_results 