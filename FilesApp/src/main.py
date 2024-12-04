import streamlit as st
import sys
import os

# Debug prints
print("Current working directory:", os.getcwd())
print("Python path:", sys.path)
print("Starting Streamlit app...")

# Bare minimum Streamlit app
st.set_page_config(page_title="Debug Mode")

def main():
    try:
        st.title("Debug Mode")
        st.write("If you can see this, basic Streamlit is working")
        st.write("Current working directory:", os.getcwd())
        
    except Exception as e:
        st.error(f"Error in main: {str(e)}")
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    try:
        print("Starting main...")
        main()
        print("Main completed")
    except Exception as e:
        print(f"Error running main: {str(e)}")
