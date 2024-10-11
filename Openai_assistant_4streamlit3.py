import streamlit as st
import sys
import pinecone
from openai import OpenAI
from pinecone import Pinecone
from pymongo import MongoClient
import uuid

# Initialize OpenAI client using Streamlit secrets
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Initialize Pinecone client
pc = Pinecone(
    api_key=st.secrets["pinecone"]["api_key"]
)

# Specify the index name
index_name = "mediation-assistant"

# Connect to the index
index = pc.Index(index_name)

# Initialize MongoDB client using Streamlit secrets
mongo_client = MongoClient(st.secrets["mongodb"]["uri"])
db = mongo_client["mediation_db"]  # Access the database
collection = db["conversations"]  # Access the 'conversations' collection

# Define mediation categories
categories = [
    "Fair Division of Assets and Debts",
    "Custody and Parenting Arrangements",
    "Financial Support (Alimony and Child Support)",
    "Minimizing Conflict and Emotional Impact",
    "Preserving Relationships",
    "Future Financial Stability",
    "Concern about your ability to negotiate with your spouse",
    "Concern about your spouse's mental and emotional stability",
    "Concern about your spouse's trustworthiness"
]

# Initialize session state variables
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'conversation_paused' not in st.session_state:
    st.session_state.conversation_paused = False
if 'conversation_ended' not in st.session_state:
    st.session_state.conversation_ended = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())  # Unique ID for each user
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'initial_question_asked' not in st.session_state:
    st.session_state.initial_question_asked = False
if 'categories_discussed' not in st.session_state:
    st.session_state.categories_discussed = []
if 'current_category' not in st.session_state:
    st.session_state.current_category = None

# Function to store conversation in MongoDB
def store_conversation(user_id, conversation):
    collection.insert_one({
        "user_id": user_id,
        "conversation": conversation
    })
    print(f"Conversation for user {user_id} stored in the database.")

# Function to upsert data into Pinecone
def upsert_to_pinecone(text, metadata):
    # Generate embedding for the text
    embedding_response = client.embeddings.create(
        input=[text],
        model='text-embedding-ada-002'
    )
    embedding = embedding_response['data'][0]['embedding']

    # Prepare data for upsert
    vector_id = str(uuid.uuid4())
    index.upsert([
        (vector_id, embedding, metadata)
    ])

# Function to retrieve relevant context from Pinecone
def retrieve_from_pinecone(query, top_k=5):
    # Generate embedding for the query
    embedding_response = client.embeddings.create(
        input=[query],
        model='text-embedding-ada-002'
    )
    query_embedding = embedding_response['data'][0]['embedding']

    # Query Pinecone index
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results['matches']

# Function to categorize user input
def categorize_user_input(user_input):
    prompt = f"""
    Given the following categories:
    - Fair Division of Assets and Debts
    - Custody and Parenting Arrangements
    - Financial Support (Alimony and Child Support)
    - Minimizing Conflict and Emotional Impact
    - Preserving Relationships
    - Future Financial Stability
    - Concern about your ability to negotiate with your spouse
    - Concern about your spouse's mental and emotional stability
    - Concern about your spouse's trustworthiness

    Please classify the following user input into one or more of these categories. Provide only the category names.

    User input:
    "{user_input}"
    """

    # Updated OpenAI API call
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant categorizing user input for divorce mediation."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0
    )

    categories_identified = response.choices[0].message.content.strip().split("\n")
    categories_identified = [cat.strip("- ").strip() for cat in categories_identified if cat.strip()]
    return categories_identified

# Function to generate interviewer responses
def generate_interviewer_response():
    # Get categories yet to be discussed
    categories_remaining = [cat for cat in categories if cat not in st.session_state.categories_discussed]

    # If all categories have been discussed, conclude the conversation
    if not categories_remaining:
        st.session_state.conversation_ended = True
        return "We've covered all the main topics that most divorcing couples consider. Is there anything else you'd like to discuss or any other concerns you'd like to share with the mediators?"

    # Set current category if not set
    if st.session_state.current_category is None:
        st.session_state.current_category = categories_remaining[0]

    # Retrieve relevant context from Pinecone
    context_matches = retrieve_from_pinecone(st.session_state.current_category)
    context_texts = [match['metadata']['text'] for match in context_matches]
    context = "\n".join(context_texts)

    instruction = f"""
    You are an AI assistant helping in divorce mediation for the Collins Family Mediation team.
    Use a friendly, non-judgmental, and supportive tone. Ask follow-up questions based on the user's input.
    Focus on the current category: {st.session_state.current_category}

    The major categories of concern are:
    {', '.join(categories)}

    Categories already discussed: {st.session_state.categories_discussed}
    Categories yet to be discussed: {categories_remaining}

    Please generate a response that explores the current category in depth. Once the category is sufficiently explored, acknowledge it and prepare to move to the next category.
    """

    messages = [{"role": "system", "content": instruction}]
    # Include the last few messages in the conversation history
    for message in st.session_state.conversation_history[-4:]:
        role = message['role']
        if role == 'user':
            messages.append({"role": "user", "content": message['content']})
        elif role == 'interviewer':
            messages.append({"role": "assistant", "content": message['content']})

    # Include retrieved context
    if context:
        messages.append({"role": "system", "content": f"Relevant context:\n{context}"})

    # Updated OpenAI API call
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# Function to generate user report
def generate_user_report(user_id, conversation):
    user_messages = "\n".join(
        [entry['content'] for entry in conversation if entry['role'] == 'user']
    )
    prompt = f"""
    Analyze the following conversation and provide a summary of the primary interests and concerns of the user, along with the apparent negotiating style:

    {user_messages}

    Provide the summary in a clear and concise manner suitable for mediation professionals.
    """

    # Updated OpenAI API call
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant analyzing a mediation conversation."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    report = response.choices[0].message.content.strip()
    return report

# Function to handle user input submission
def submit_response():
    if st.session_state.user_input:
        user_input = st.session_state.user_input
        # Append user input to conversation history
        st.session_state.conversation_history.append({"role": "user", "content": user_input})

        # Upsert user input into Pinecone
        upsert_to_pinecone(user_input, {"text": user_input, "role": "user"})

        # Categorize user input
        categories_identified = categorize_user_input(user_input)
        for category in categories_identified:
            if category not in st.session_state.categories_discussed:
                st.session_state.categories_discussed.append(category)

        # Check if current category is completed
        if st.session_state.current_category in categories_identified:
            # Assume topic is completed after user's input
            st.session_state.categories_discussed.append(st.session_state.current_category)
            st.session_state.current_category = None

        # Generate interviewer response
        interviewer_response = generate_interviewer_response()
        st.session_state.conversation_history.append({"role": "interviewer", "content": interviewer_response})

        # Upsert interviewer response into Pinecone
        upsert_to_pinecone(interviewer_response, {"text": interviewer_response, "role": "assistant"})

        # Clear the input by setting it to an empty string
        st.session_state.user_input = ""

# Main app functionality
def main():
    st.title("Collins Family Mediation Interviewer")

    if st.session_state.user_name is None:
        st.write("I'm here to gather information to help the Collins Family Mediation team.")
       
