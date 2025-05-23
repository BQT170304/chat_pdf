from typing import List, Dict, Any, Optional
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint

from opensearch_client import OpenSearchClient

class RAGPipeline:
    """RAG Pipeline for generating responses based on retrieved contexts"""
    
    def __init__(self, opensearch_client: OpenSearchClient):
        """Initialize RAG Pipeline"""
        self.opensearch_client = opensearch_client
        
        # Initialize the LLM - use OpenAI if API key is available, otherwise fall back to HuggingFace
        if os.environ.get("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
        else:
            # Use Hugging Face LLM
            self.llm = HuggingFaceEndpoint(
                repo_id="google/flan-t5-large",
                temperature=0.2,
                max_length=512
            )
        
        # Setup the prompt template
        self.prompt = ChatPromptTemplate.from_template("""
        Answer the following question based on the provided context.
        If you don't know the answer or can't find it in the context, just say "I don't have enough information to answer this question."
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """)
        
        # Create the RAG chain
        self.rag_chain = (
            {"context": self._retrieve_context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _retrieve_context(self, query: str, search_type: str = "hybrid") -> str:
        """
        Retrieve context from OpenSearch
        
        Args:
            query: User query
            search_type: Type of search to perform
            
        Returns:
            Combined context from relevant chunks
        """
        # Get chunks from OpenSearch
        chunks = self.opensearch_client.search(query, search_type=search_type, size=5)
        
        # Combine the chunks into a single context string
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"Chunk {i+1}:\n"
            context += f"Title: {chunk['title']}\n"
            context += f"Content: {chunk['content']}\n\n"
        
        return context if context else "No relevant information found."
    
    def generate_response(self, query: str, search_type: str = "hybrid") -> str:
        """
        Generate a response to the user's query using RAG
        
        Args:
            query: User query
            search_type: Type of search to perform (semantic, keyword, or hybrid)
            
        Returns:
            Generated response
        """
        # Store the search type in the instance for _retrieve_context to use
        self._current_search_type = search_type
        
        # Get response from the RAG chain
        try:
            response = self.rag_chain.invoke(query)
            return response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request." 