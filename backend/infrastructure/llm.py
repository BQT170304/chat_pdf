import os
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint

from backend.domain.interfaces import LLMInterface
from backend.domain.entities import Chunk


class LangChainOpenAILLM(LLMInterface):
    """LLM implementation using LangChain with OpenAI models"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.2):
        """Initialize the LLM with a model name"""
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self.prompt = None
        self.rag_chain = None
    
    def initialize(self) -> None:
        """Initialize the LLM and create the prompt template"""
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        
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
        
        # Create the chain
        self.rag_chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _prepare_context(self, chunks: List[Chunk]) -> str:
        """Prepare context from chunks"""
        if not chunks:
            return "No relevant information found."
        
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"Chunk {i+1}:\n"
            context += f"Title: {chunk.title}\n"
            context += f"Content: {chunk.content}\n\n"
        
        return context
    
    def generate_response(self, query: str, context: List[Chunk] = None) -> str:
        """Generate a response to a query, optionally using context"""
        self.initialize()
        
        try:
            context_str = self._prepare_context(context) if context else "No context provided."
            
            # Invoke the chain
            response = self.rag_chain.invoke({
                "context": context_str,
                "question": query
            })
            
            return response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"I'm sorry, I encountered an error while processing your request: {str(e)}"


class HuggingFaceLLM(LLMInterface):
    """LLM implementation using LangChain with HuggingFace models"""
    
    def __init__(self, repo_id: str = "google/flan-t5-large", temperature: float = 0.2):
        """Initialize the LLM with a model name"""
        self.repo_id = repo_id
        self.temperature = temperature
        self.llm = None
        self.prompt = None
        self.rag_chain = None
    
    def initialize(self) -> None:
        """Initialize the LLM and create the prompt template"""
        self.llm = HuggingFaceEndpoint(
            repo_id=self.repo_id,
            temperature=self.temperature,
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
        
        # Create the chain
        self.rag_chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _prepare_context(self, chunks: List[Chunk]) -> str:
        """Prepare context from chunks"""
        if not chunks:
            return "No relevant information found."
        
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"Chunk {i+1}:\n"
            context += f"Title: {chunk.title}\n"
            context += f"Content: {chunk.content}\n\n"
        
        return context
    
    def generate_response(self, query: str, context: List[Chunk] = None) -> str:
        """Generate a response to a query, optionally using context"""
        self.initialize()
        
        try:
            context_str = self._prepare_context(context) if context else "No context provided."
            
            # Invoke the chain
            response = self.rag_chain.invoke({
                "context": context_str,
                "question": query
            })
            
            return response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"I'm sorry, I encountered an error while processing your request: {str(e)}" 