from pinecone import Pinecone
from openai import OpenAI
from typing import Dict, List, Optional

class VectorStore:
    def __init__(self, pinecone_index):
        self.pinecone_index = pinecone_index
        self.client = OpenAI()
        self.similarity_threshold = 0.7

    def query(self, 
              text: str, 
              context_window: Optional[str] = None,
              top_k: int = 5) -> Dict:
        try:
            # Generate embedding for current input
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            embedding = response.data[0].embedding

            # If context window provided, also get its embedding
            if context_window:
                context_response = self.client.embeddings.create(
                    model="text-embedding-3-large",
                    input=context_window
                )
                context_embedding = context_response.data[0].embedding
                
                # Query both current input and context
                results = self._query_with_context(embedding, context_embedding, top_k)
            else:
                results = self.pinecone_index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True
                )

            return self._process_results(results)

        except Exception as e:
            print(f"Error in vector store query: {str(e)}")
            return {}

    def _query_with_context(self, 
                          current_embedding: List[float], 
                          context_embedding: List[float],
                          top_k: int) -> Dict:
        # Query with both embeddings and combine results
        current_results = self.pinecone_index.query(
            vector=current_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        context_results = self.pinecone_index.query(
            vector=context_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Combine and deduplicate results
        return self._merge_results(current_results, context_results)

    def _merge_results(self, current_results: Dict, context_results: Dict) -> Dict:
        # Implement logic to merge and deduplicate results
        # Prioritize based on both similarity and metadata
        # Note: This is a placeholder - implementation needed
        return current_results

    def _process_results(self, results: Dict) -> Dict:
        processed_info = {
            'question_strategies': [],
            'mediation_topics': [],
            'emotional_support': [],
            'high_priority': []
        }
        
        for match in results["matches"]:
            if match.score < self.similarity_threshold:
                continue
                
            metadata = match.get('metadata', {})
            content = metadata.get('content', '')
            category = metadata.get('category1', '')
            priority = metadata.get('priority', 'normal')
            
            if priority == 'high':
                processed_info['high_priority'].append(content)
                
            if category in processed_info:
                processed_info[category].append(content)
                
        return processed_info
