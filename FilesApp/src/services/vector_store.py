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
              conversation_context: Optional[Dict] = None,
              top_k: int = 5) -> Dict:
        try:
            # Generate embedding for current input
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            embedding = response.data[0].embedding

            # Build filter based on conversation context
            filter_dict = self._build_filter(conversation_context) if conversation_context else {}

            # Query Pinecone with filter
            results = self.pinecone_index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )

            return self._process_results(results)

        except Exception as e:
            print(f"Error in vector store query: {str(e)}")
            return {}

    def _build_filter(self, context: Dict) -> Dict:
        """Build Pinecone filter based on conversation context"""
        filter_dict = {}
        
        if context.get('mediation_phase'):
            filter_dict['mediation_phase'] = {'$eq': context['mediation_phase']}
        
        if context.get('topic_domain'):
            filter_dict['topic_domain'] = {'$eq': context['topic_domain']}
        
        # Add emotional context filtering
        if context.get('emotion_sensitive'):
            filter_dict['context_flags.emotion_sensitive'] = {'$eq': True}
        
        # Handle specific content needs
        if context.get('needs_calculation'):
            filter_dict['content_type'] = {'$eq': 'Calculation_Guide'}
        
        return filter_dict

    def _process_results(self, results: Dict) -> Dict:
        processed_info = {
            'high_priority': [],
            'phase_specific': [],
            'topic_specific': [],
            'emotional_support': [],
            'technical_info': []
        }
        
        for match in results["matches"]:
            if match.score < self.similarity_threshold:
                continue
                
            metadata = match.get('metadata', {})
            content = metadata.get('chunk_text', '')
            
            # Organize by interaction style and priority
            if metadata.get('priority') == 'High':
                processed_info['high_priority'].append({
                    'content': content,
                    'style': metadata.get('interaction_style'),
                    'phase': metadata.get('mediation_phase')
                })
            
            # Phase-specific information
            if metadata.get('mediation_phase'):
                processed_info['phase_specific'].append({
                    'content': content,
                    'phase': metadata.get('mediation_phase')
                })

            # Topic-specific information
            if metadata.get('topic_domain'):
                processed_info['topic_specific'].append({
                    'content': content,
                    'domain': metadata.get('topic_domain')
                })

            # Emotional support content
            if metadata.get('context_flags', {}).get('emotion_sensitive'):
                processed_info['emotional_support'].append({
                    'content': content,
                    'style': metadata.get('interaction_style')
                })

            # Technical information
            if metadata.get('interaction_style') == 'Technical':
                processed_info['technical_info'].append({
                    'content': content,
                    'domain': metadata.get('topic_domain')
                })
                
        return processed_info
