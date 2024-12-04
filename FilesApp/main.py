import streamlit as st
import os
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def main():
    st.title("Debug Mode")
    st.write("If you can see this, basic Streamlit is working")
    st.write("Current working directory:", os.getcwd())

if __name__ == "__main__":
    main()
