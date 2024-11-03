import streamlit as st
from transformers import pipeline
import re
from datetime import datetime
import json

class ChatHistory:
    def __init__(self):
        self.history = []
    
    def add_message(self, role, content):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M")
        })
    
    def get_history(self):
        return self.history

class ProblemSolver:
    def __init__(self):
        self.model = pipeline("text-generation", 
                            model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
    
    def generate_solution(self, problem_description):
        prompt = f"""Given this programming problem:
        {problem_description}
        
        Please provide an optimized Python solution with:
        1. Time and space complexity analysis
        2. Clear code comments
        3. Example test cases
        4. Edge case handling
        
        Format the solution in Python code."""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.model(messages)
        
        # Extract Python code from the response
        solution = self.extract_code(response[0]['generated_text'])
        return solution
    
    @staticmethod
    def extract_code(text):
        # Extract code between triple backticks if present
        code_pattern = r"```python(.*?)```"
        matches = re.findall(code_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        return text.strip()

def main():
    st.set_page_config(page_title="Quantitative Problem Solver", layout="wide")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ChatHistory()
    if 'solver' not in st.session_state:
        st.session_state.solver = ProblemSolver()
    
    # App title and description
    st.title("💻 Quantitative Problem Solver")
    st.markdown("""
    Get optimized Python solutions for your programming problems.
    Just paste the problem description or link below!
    """)
    
    # Create two columns for chat interface
    chat_col, info_col = st.columns([2, 1])
    
    with chat_col:
        # Chat interface
        st.subheader("Chat Interface")
        
        # Display chat history
        for message in st.session_state.chat_history.get_history():
            with st.chat_message(message["role"]):
                st.write(f"**{message['timestamp']}**")
                if message["role"] == "assistant" and "```python" in message["content"]:
                    st.code(message["content"], language="python")
                else:
                    st.write(message["content"])
        
        # User input
        user_input = st.chat_input("Enter your programming problem or link...")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.add_message("user", user_input)
            
            # Generate solution
            try:
                solution = st.session_state.solver.generate_solution(user_input)
                st.session_state.chat_history.add_message("assistant", solution)
                
                # Show the latest response
                with st.chat_message("assistant"):
                    st.code(solution, language="python")
            except Exception as e:
                st.error(f"Error generating solution: {str(e)}")
    
    with info_col:
        # Information panel
        st.subheader("Solution Stats")
        if st.session_state.chat_history.get_history():
            latest_solution = st.session_state.chat_history.get_history()[-1]
            if latest_solution["role"] == "assistant":
                code = latest_solution["content"]
                st.metric("Lines of Code", len(code.split("\n")))
                st.metric("Characters", len(code))
        
        # Export chat history
        if st.button("Export Chat History"):
            chat_export = json.dumps(st.session_state.chat_history.get_history(), 
                                   indent=2)
            st.download_button(
                label="Download Chat History",
                data=chat_export,
                file_name="chat_history.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
