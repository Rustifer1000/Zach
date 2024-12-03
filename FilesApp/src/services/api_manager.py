from openai import OpenAI
from typing import Dict, List
import json

class APIManager:
    def __init__(self, pinecone_index):
        self.pinecone_index = pinecone_index
        self.client = OpenAI()

    def query_context(self, user_input: str, conversation_history: List[Dict]) -> str:
        """Enhanced context querying with conversation history"""
        try:
            # Get context from both immediate input and recent conversation
            context_window = " ".join([msg["content"] for msg in conversation_history[-3:]])
            
            # Query both contexts
            immediate_context = self._query_pinecone(user_input, 0.75)
            conversation_context = self._query_pinecone(context_window, 0.7)
            
            # Combine contexts with relevant categories
            categories = self._analyze_relevant_categories(user_input)
            
            return f"""Consider this relevant information when responding:

            Immediate Context:
            {immediate_context}
            
            Conversation Context:
            {conversation_context}
            
            Relevant Categories: {', '.join(categories)}
            """

        except Exception as e:
            print(f"Error in query_context: {str(e)}")
            return ""

    def _query_pinecone(self, text: str, similarity_threshold: float = 0.7) -> str:
        """Internal method for querying Pinecone with enhanced metadata handling"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            
            embedding = response.data[0].embedding
            
            results = self.pinecone_index.query(
                vector=embedding,
                top_k=5,
                include_metadata=True
            )

            # Organize results by priority
            categorized_results = {
                'high': [],
                'medium': [],
                'low': []
            }

            for match in results["matches"]:
                if match.score < similarity_threshold:
                    continue
                    
                metadata = match.get('metadata', {})
                priority = metadata.get('priority', 'low')
                
                # Format result with complete metadata
                result = f"""
                Title: {metadata.get('title', 'Untitled')}
                Categories: {metadata.get('category1', '')} {f"/ {metadata.get('category2', '')}" if metadata.get('category2') else ''}
                Author: {metadata.get('author', 'Unknown')}
                Date: {metadata.get('date', 'Unspecified')}
                Content: {metadata.get('content', '')}
                Priority: {priority.upper()}
                Relevance Score: {match.score:.2f}
                """
                
                categorized_results[priority].append(result)

            # Combine results in priority order
            all_results = []
            for priority in ['high', 'medium', 'low']:
                if categorized_results[priority]:
                    all_results.extend(categorized_results[priority])
            
            return "\n\n".join(all_results[:3])  # Return top 3 most relevant results

        except Exception as e:
            print(f"Error in _query_pinecone: {str(e)}")
            return ""

    def _analyze_relevant_categories(self, text: str) -> List[str]:
        """Analyze text to identify relevant document categories"""
        categories = []
        keywords = {
            'parenting': ['child', 'parent', 'custody', 'visitation', 'kids'],
            'support': ['support', 'assistance', 'help', 'aid', 'alimony'],
            'assets': ['property', 'money', 'financial', 'assets', 'house', 'car'],
            'legal': ['court', 'legal', 'law', 'attorney', 'lawyer', 'divorce'],
            'trust': ['trust', 'confidence', 'reliable', 'honest'],
            'potential conflict': ['dispute', 'conflict', 'disagreement', 'fight', 'argument']
        }
        
        for category, words in keywords.items():
            if any(word.lower() in text.lower() for word in words):
                categories.append(category)
        
        return categories[:2]

    def generate_response(self, messages: List[Dict]) -> str:
        """Generate AI response using the provided context"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return "I apologize, but I encountered an error processing your request."
        return "\n".join(formatted_sections)
