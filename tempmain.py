import os
import streamlit as st
import openai
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
import json
import pinecone
import uuid

# Ensure OpenAI and Pinecone API keys are set via st.secrets or environment variables
openai.api_key = st.secrets["openai"]["api_key"]

# ---- Constants and Configuration ----
class ModelTier(Enum):
    ANALYSIS = "analysis"
    RESPONSE = "response"
    EMBEDDING = "embedding"

@dataclass
class ModelConfig:
    name: str
    tokens_per_second: float
    cost_per_1k_tokens: float
    max_context_tokens: int
    latency_ms: float
    quality_score: float

MODEL_CONFIGS = {
    "gpt-4o": ModelConfig("gpt-4o", 12, 0.03, 3072, 2000, 1.0),
    "gpt-4o": ModelConfig("gpt-4o", 30, 0.0015, 16385, 500, 0.85),
    "text-embedding-3-large": ModelConfig("text-embedding-3-large", 100, 0.0004, 3072, 100, 0.90)
}

INITIAL_GREETING = """Hello! I'm the Collins Family Mediation Intermediary. I'm here to gather information and clarify issues to help you get a head start on your mediation sessions with the Collinses. To begin, could you please tell me your first name?"""

SYSTEM_MESSAGE = """
You are an AI intermediary for Collins Family Mediation. Your role is to gather detailed, nuanced information ... (same instructions as previously provided) ...
"""

class APIManager:
    def __init__(self, pinecone_index):
        self.pinecone_index = pinecone_index

    def embed_text(self, text: str) -> List[float]:
        response = openai.Embedding.create(
            model="text-embedding-3-large",
            input=text
        )
        return response["data"][0]["embedding"]

    def query_pinecone(self, user_input: str) -> str:
        try:
            embedding = self.embed_text(user_input)

            # Query Pinecone
            results = self.pinecone_index.query(
                vector=embedding,
                top_k=3,
                include_metadata=True
            )

            relevant_info = []
            for match in results["matches"]:
                metadata = match.get('metadata', {})
                info_parts = []
                # Include fields from both documents and conversation turns
                if "title" in metadata:
                    info_parts.append(f"Title: {metadata['title']}")
                if "category1" in metadata:
                    info_parts.append(f"Category1: {metadata['category1']}")
                if "category2" in metadata and metadata['category2']:
                    info_parts.append(f"Category2: {metadata['category2']}")
                if "priority" in metadata:
                    info_parts.append(f"Priority: {metadata['priority']}")
                if "user_id" in metadata:
                    info_parts.append(f"User ID: {metadata['user_id']}")
                if "snippet" in metadata:
                    info_parts.append(f"Snippet: {metadata['snippet']}")
                if "role" in metadata:
                    info_parts.append(f"Role: {metadata['role']}")
                if "type" in metadata:
                    info_parts.append(f"Type: {metadata['type']}")

                if info_parts:
                    relevant_info.append("\n".join(info_parts))

            return "\n\n".join(relevant_info) if relevant_info else ""
        except Exception as e:
            print(f"Error in query_pinecone: {str(e)}")
            return ""

    def generate_response(self, messages: List[Dict]) -> str:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return "I apologize, but I encountered an error processing your request."

    def store_conversation_turn(self, user_id: str, conversation_id: str, role: str, content: str):
        # Embed and store a single conversation turn as a vector
        embeddings = self.embed_text(content)
        doc_id = f"conversation_{conversation_id}_{role}_{uuid.uuid4().hex[:6]}"
        metadata = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "role": role,
            "snippet": content,
            "type": "conversation"
        }
        self.pinecone_index.upsert(vectors=[(doc_id, embeddings, metadata)])


def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True

        # Prompt user for user_id
        if "user_id" not in st.session_state:
            st.session_state.user_id = st.text_input("Please enter your user ID:", key="init_user_id")
            st.stop()

        if st.session_state.user_id:
            # Generate a unique conversation_id for this session
            if "conversation_id" not in st.session_state:
                st.session_state.conversation_id = uuid.uuid4().hex[:8]

            st.session_state.messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "assistant", "content": INITIAL_GREETING}
            ]
            st.session_state.current_response = INITIAL_GREETING
            st.session_state.backend_messages = []


def main():
    st.title("Collins Family Mediation AI Intermediary")

    # Initialize session state and wait for user_id
    initialize_session_state()
    if not st.session_state.get("user_id"):
        return

    user_id = st.session_state.user_id
    conversation_id = st.session_state.conversation_id

    # Initialize Pinecone
    pinecone.init(
        api_key=st.secrets["pinecone"]["api_key"],
        environment=st.secrets["pinecone"]["environment"]
    )
    index = pinecone.Index("mediation4")
    api_manager = APIManager(index)

    # Display the AI response
    st.write(st.session_state.current_response)

    user_input = st.chat_input("Your response:")

    if user_input:
        # Store user message in conversation memory
        api_manager.store_conversation_turn(user_id, conversation_id, "user", user_input)

        # Add user message to conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Query Pinecone for relevant info (including past conversation turns and documents)
        relevant_info = api_manager.query_pinecone(user_input)
        if relevant_info:
            st.session_state.backend_messages.append({
                "role": "system",
                "content": f"Relevant context:\n{relevant_info}"
            })

        # Combine messages for API call
        full_context = (
            [st.session_state.messages[0]]  # system message
            + st.session_state.backend_messages
            + st.session_state.messages[1:]
        )

        # Generate response
        response = api_manager.generate_response(full_context)
        st.session_state.current_response = response
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Store assistant message in conversation memory
        api_manager.store_conversation_turn(user_id, conversation_id, "assistant", response)

        # Refresh display
        st.experimental_rerun()

if __name__ == "__main__":
    main()
