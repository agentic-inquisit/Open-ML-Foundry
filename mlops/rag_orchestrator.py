import os
from .embedding_service import VisualEmbeddingService
from pymilvus import connections, Collection

from langchain_community.llms import LlamaCpp
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

LLAMACPP_MODEL_PATH = os.getenv("LLAMACPP_MODEL_PATH")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")


def _load_llm():
    if not LLAMACPP_MODEL_PATH:
        raise RuntimeError(
            "LLAMACPP_MODEL_PATH not set — point it at a local GGUF model file "
            "for RAGOrchestrator's summarization step."
        )
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    return LlamaCpp(
        model_path=LLAMACPP_MODEL_PATH,
        temperature=0.7,
        max_tokens=2000,
        top_p=0.95,
        callback_manager=callback_manager,
        verbose=True,
    )


class RAGOrchestrator:
    def __init__(self):
        self.embedder = VisualEmbeddingService()
        self.llm = _load_llm()
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        # Assumes a "visual_events" collection already exists in Milvus with an
        # "embedding" vector field plus "timestamp"/"stream_id"/"metadata"
        # scalar fields — this module has no code path that creates or
        # populates that collection.
        self.collection = Collection("visual_events")

    async def search_events(self, query_text, top_k=5):
        """
        Perform semantic search for visual events.
        """
        query_vector = self.embedder.get_text_embedding(query_text)

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            [query_vector], "embedding", search_params, limit=top_k,
            output_fields=["timestamp", "stream_id", "metadata"],
        )
        # pymilvus returns Hit objects, not dicts — normalize here so callers
        # (generate_response below) can use plain dict access.
        return [
            {
                "timestamp": hit.entity.get("timestamp"),
                "stream_id": hit.entity.get("stream_id"),
                "metadata": hit.entity.get("metadata"),
            }
            for hits in results for hit in hits
        ]

    async def generate_response(self, query_text):
        """
        RAG Flow: Retrieve relevant events and generate a summary using an LLM.
        """
        events = await self.search_events(query_text)

        context = "\n".join(
            f"Time: {e['timestamp']}, Stream: {e['stream_id']}, Info: {e['metadata']}"
            for e in events
        )
        prompt = f"Based on the following visual events found in the stream:\n{context}\n\nAnswer the user: {query_text}"

        return self.llm(prompt)
