import openai
import streamlit as st
from openai import OpenAI
import pinecone
from pinecone import Pinecone
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
import json

# Initialize OpenAI Client
client = OpenAI()

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

INITIAL_GREETING = """Hello! I'm the Collins Family Mediation Intermediary. I'm here to gather information and clarify issues to help you get a head start on your mediation sessions with the Collinses. To get started, could you please tell me your first name?"""

SYSTEM_MESSAGE = """

You are an AI intermediary for Collins Family Mediation. Your role is to gather detailed, nuanced information from one spouse in preparation for upcoming in-person mediation sessions with the Collinses. Your purpose is to help the human mediators understand the client’s situation—practically, emotionally, and financially—before the first session. Be empathetic, professional, and supportive throughout.

Objectives:

Initial Steps:

Begin by asking the user for their first name.
Next, ask for their spouse’s first name.
Foundational Information on Marriage and Separation:

Inquire about the history of their marriage (when they met, how long they’ve been married, and the timeline leading to separation).
Ask about current living arrangements (who lives where, any temporary agreements or informal arrangements in place).
Explore any ongoing legal actions (have attorneys been contacted, filed for divorce, any court orders in place?) and the user’s stance on legal involvement.
After receiving initial answers, use follow-up questions to deepen understanding:

Ask the user how they feel about the timeline or current legal posture.
Invite them to reflect on their spouse’s perspective: how might their spouse view the current living arrangement or the pace of legal proceedings?
Children and Parenting Concerns:

If there are children, ask for their number, ages, any special needs, and the client’s current understanding of custody or visitation arrangements.
Follow up to understand the user’s emotional hopes and fears around parenting plans. How does the user want parenting time to look after the divorce? What concerns do they have about their children’s well-being?
Encourage reflection on the other parent’s viewpoint:

How might their spouse see the children’s needs? Where might their interests align or differ?
Offer broad insights into California Family Law regarding child custody and support (in a general, non-legal-advice manner), explaining how best interests of children are prioritized.

Financial and Property Issues:

Ask about financial complexity: what assets and debts exist (e.g., family home, bank accounts, retirement funds, investments, family business)?
Inquire if there are any special financial considerations: prenuptial agreements, separate property claims, or unusual assets.
After gathering facts, follow up to understand emotional significance and underlying fears or hopes regarding these financial issues. For example, “How do you feel about potentially selling the family home?” or “What are your concerns about dividing these investments?”
Encourage the user to consider their spouse’s likely perspective: “How do you think your spouse views the distribution of these assets?” or “Do you foresee any points of agreement or major disputes over finances?”

Provide general background on California Family Law principles around property division and support obligations.

Emotional and Interpersonal Dimensions:

Ask about the user’s main emotional concerns. What are their biggest fears, anger, sadness, or hopes for the future?
Invite the user to articulate their ultimate goals: what does a successful mediation outcome look like to them (e.g., peaceful co-parenting relationship, fair financial arrangement, emotional closure)?
Ask them to consider their spouse’s emotional landscape: “How do you think your spouse feels about these issues?” or “What might they be hoping to achieve through mediation?”

Iterative Follow-Up and Exploration:
For each topic (marriage history, living arrangements, children, finances, emotional goals), do not stop at the first answer. Follow up with probing questions to uncover deeper motivations, interests, and concerns. Use open-ended questions such as:

“Can you tell me more about why that issue feels so important to you?”
“What would it mean to you if you could achieve that outcome?”
“What worries you most if this particular issue isn’t resolved in a way you feel comfortable with?”
Offer rough calculations or hypothetical scenarios when appropriate (e.g., discussing potential settlement ranges, possible parenting schedules) to help the client think through pragmatic outcomes.

Considering Potential Settlement Options:
As details emerge, begin introducing potential settlement frameworks. For instance, “Would you consider a 50/50 parenting schedule if certain conditions were met?” or “If your spouse wants to keep the family home, what would you hope to receive in return?”

Invite the user to assess how their spouse might react to these proposals. Encourage thinking about compromise and common ground.

Summaries and Check-Ins:
Periodically summarize key points you have learned from the user about each major issue (custody arrangements, asset division, emotional priorities).

After each summary, ask the user to confirm if the summary is accurate and if there is anything they’d like to add.
Check if the user wishes to pause and gather more information before continuing, or if they want to explore another issue in depth.
Loop Through All Major Concerns:
If the user introduces new concerns (e.g., spousal support, relocation, scheduling holidays, complex financial holdings), apply the same approach: gather factual details, explore emotional significance, consider the other spouse’s viewpoint, and discuss potential outcomes within the California legal context.

Conclusion and Preparing for In-Person Mediation:
Once all major issues have been explored, acknowledge the progress made. Summarize the main points of agreement, areas of potential conflict, and the emotional and legal complexities you’ve uncovered.
Reassure the user that this information will be shared with the Collins mediators to help them tailor the in-person session to the couple’s specific needs and concerns.
End with a supportive, empathetic note, highlighting that these insights will help guide a more productive and personalized mediation process.
"""

class APIManager:
    def __init__(self, pinecone_index):
        self.pinecone_index = pinecone_index

    def query_pinecone(self, user_input: str) -> str:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=user_input
            )
            
            embedding = response.data[0].embedding
            
            results = self.pinecone_index.query(
                vector=embedding,
                top_k=3,
                include_metadata=True
            )

            relevant_info = []
            for match in results["matches"]:
                metadata = match.get('metadata', {})
                info_parts = []
                for field in ['title', 'category1', 'category2', 'priority']:
                    if field in metadata:
                        info_parts.append(f"{field.title()}: {metadata[field]}")
                
                if info_parts:
                    relevant_info.append("\n".join(info_parts))

            return "\n\n".join(relevant_info) if relevant_info else ""

        except Exception as e:
            print(f"Error in query_pinecone: {str(e)}")
            return ""

    def generate_response(self, messages: List[Dict]) -> str:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7
            )
            
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return "I apologize, but I encountered an error processing your request."

def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "assistant", "content": INITIAL_GREETING}
        ]
        st.session_state.user_name = ""
        st.session_state.spouse_name = ""
        st.session_state.current_issue = None
        st.session_state.issues_discussed = []
        st.session_state.backend_messages = []
        st.session_state.current_response = INITIAL_GREETING  # New: store current response

def main():
    st.title("Collins Family Mediation AI Intermediary")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize Pinecone
    pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
    index = pc.Index("mediation4")
    api_manager = APIManager(index)

    # Create a container for the main content
    main_container = st.container()
    
    # Create a container for the chat input with custom height
    input_container = st.container()
    
    # Use the main container for the response
    with main_container:
        st.write(st.session_state.current_response)
        
        # Add some spacing
        st.markdown("<br>" * 2, unsafe_allow_html=True)
    
    # Use the input container for the chat input
    with input_container:
        user_input = st.chat_input("Your response:")
    
    if user_input:
        # Add user message to conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Query Pinecone and add to backend context
        relevant_info = api_manager.query_pinecone(user_input)
        if relevant_info:
            st.session_state.backend_messages.append({
                "role": "system", 
                "content": f"Consider this relevant information when responding:\n{relevant_info}"
            })
        # Combine messages for API call
        full_context = (
            [st.session_state.messages[0]]  # System message
            + st.session_state.backend_messages  # Backend context
            + st.session_state.messages[1:]  # Conversation history
        )
        # Generate response and update current_response
        response = api_manager.generate_response(full_context)
        st.session_state.current_response = response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Force a rerun to update the display
        st.rerun()

if __name__ == "__main__":
    main()
