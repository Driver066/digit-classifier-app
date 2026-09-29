import streamlit as st

st.title("Interactive Input App 💬")

# Create a text input widget
user_text = st.text_input("Type something here:")

# Display the output when text is entered
if user_text:
    st.write(f"You typed: **{user_text}**")

