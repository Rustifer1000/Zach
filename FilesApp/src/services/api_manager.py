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
        """Get relevant context based on user input and conversation history"""
        
        # Extract conversation context
        context = self._analyze_conversation_context(conversation_history)
        
        # Query vector store with context
        relevant_info = self.vector_store.query(
            text=user_input,
            conversation_context=context
        )
        
        return self._format_context(relevant_info)

    def _analyze_conversation_context(self, conversation_history: List[Dict]) -> Dict:
        """Analyze conversation to determine context and phase"""
        if not conversation_history:
            return {'mediation_phase': 'Initial_Contact'}

        # Get recent messages
        recent_messages = conversation_history[-3:]
        
        context = {
            'mediation_phase': self._determine_phase(recent_messages),
            'emotion_sensitive': self._check_emotional_content(recent_messages),
            'needs_calculation': self._check_calculation_needed(recent_messages)
        }

        # Extract topic if present
        topic = self._extract_topic(recent_messages)
        if topic:
            context['topic_domain'] = topic

        return context

    def _determine_phase(self, messages: List[Dict]) -> str:
        # Add logic to determine current mediation phase
        # This is a simplified example
        return 'Initial_Contact' if len(messages) < 3 else 'Issue_Identification'

    def _check_emotional_content(self, messages: List[Dict]) -> bool:
        # Check if recent messages indicate emotional content
        emotional_keywords = ['feel', 'upset', 'angry', 'worried', 'concerned']
        return any(any(keyword in msg['content'].lower() for keyword in emotional_keywords) 
                  for msg in messages)

    def _check_calculation_needed(self, messages: List[Dict]) -> bool:
        # Check if calculations might be needed
        calculation_keywords = ['calculate', 'amount', 'payment', 'support', 'assets']
        return any(any(keyword in msg['content'].lower() for keyword in calculation_keywords) 
                  for msg in messages)

    def _extract_topic(self, messages: List[Dict]) -> Optional[str]:
        # Extract main topic from recent messages
        # Add logic to map conversation content to topic domains
        return None  # Placeholder

    def _format_context(self, relevant_info: Dict) -> str:
        """Format the retrieved information for the LLM context"""
        formatted_sections = []
        
        if relevant_info['high_priority']:
            formatted_sections.append("HIGH PRIORITY GUIDANCE:")
            for item in relevant_info['high_priority']:
                formatted_sections.append(f"- {item['content']}")

        if relevant_info['phase_specific']:
            formatted_sections.append("\nPHASE-SPECIFIC GUIDANCE:")
            for item in relevant_info['phase_specific']:
                formatted_sections.append(f"- {item['content']}")

        if relevant_info['emotional_support']:
            formatted_sections.append("\nEMOTIONAL SUPPORT GUIDANCE:")
            for item in relevant_info['emotional_support']:
                formatted_sections.append(f"- {item['content']}")

        if relevant_info['technical_info']:
            formatted_sections.append("\nTECHNICAL INFORMATION:")
            for item in relevant_info['technical_info']:
                formatted_sections.append(f"- {item['content']}")

        return "\n".join(formatted_sections)
