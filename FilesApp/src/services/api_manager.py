from openai import OpenAI
from typing import Dict, List
from .vector_store import VectorStore

class APIManager:
    def __init__(self, pinecone_index):
        self.vector_store = VectorStore(pinecone_index)
        self.client = OpenAI()

    def query_context(self, 
                     user_input: str, 
                     conversation_history: List[Dict]) -> str:
        # Get recent conversation context
        context_window = self._get_context_window(conversation_history)
        
        # Query vector store with both current input and context
        relevant_info = self.vector_store.query(
            text=user_input,
            context_window=context_window
        )
        
        return self._format_context(relevant_info)

    def generate_response(self, messages: List[Dict]) -> str:
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

    def _get_context_window(self, conversation_history: List[Dict], window_size: int = 3) -> str:
        recent_messages = conversation_history[-window_size:]
        return " ".join([msg["content"] for msg in recent_messages])

    def _format_context(self, relevant_info: Dict) -> str:
        formatted_sections = []
        
        if relevant_info['high_priority']:
            formatted_sections.append("HIGH PRIORITY INFORMATION:")
            formatted_sections.extend(relevant_info['high_priority'])
            
        for category, items in relevant_info.items():
            if category != 'high_priority' and items:
                formatted_sections.append(f"\n{category.upper()}:")
                formatted_sections.extend(items)
                
        return "\n".join(formatted_sections)
