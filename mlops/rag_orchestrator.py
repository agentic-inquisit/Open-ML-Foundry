import os
from .embedding_service import VisualEmbeddingService
# Assuming a generic VectorDB client and LLM client
# from pymilvus import connections, Collection
# import openai

from langchain.llms import LlamaCpp
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

model_path = "/path/" 

llm = LlamaCpp(
            model_path=model_path,
            temperature=0.7,
            max_tokens=2000,
            top_p=0.95,
            callback_manager=callback_manager,
            verbose=True,  # Verbose output
        )

class RAGOrchestrator:
    def __init__(self):
        self.embedder = VisualEmbeddingService()
        # Initialize VectorDB connection
        connections.connect("default", host="localhost", port="19530")
        self.collection = Collection("visual_events")

    async def search_events(self, query_text, top_k=5):
        """
        Perform semantic search for visual events.
        """
        query_vector = self.embedder.get_text_embedding(query_text)
         
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search([query_vector], "embedding", search_params, limit=top_k)
        
        return results

    async def generate_response(self, query_text):
        """
        RAG Flow: Retrieve relevant events and generate a summary using an LLM.
        """
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]) 
        events = await self.search_events(query_text)
         
        context = "\n".join([f"Time: {e['timestamp']}, Stream: {e['stream_id']}, Info: {e['metadata']}" for e in events])
        prompt = f"Based on the following visual events found in the stream:\n{context}\n\nAnswer the user: {query_text}"
         
        response = llm(prompt)
        return response.choices[0].message.content
        
        return f"Identified {len(events)} relevant events. Summary: {context}"
