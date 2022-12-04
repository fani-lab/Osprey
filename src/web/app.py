import streamlit as st
from streamlit_chat import message
import requests
import urllib

st.set_page_config(
    page_title="Predatory Conversation Detection - Demo",
    page_icon=":robot:"
)

#api
headers = {
	"X-RapidAPI-Key": "ca2df21b0cmsh1e3a19211af1628p147906jsn2171b437902e",
	"X-RapidAPI-Host": "chatbot-chatari.p.rapidapi.com"
}

st.header("Predatory Conversation Detection - Demo")
st.markdown("[Github](https://github.com/fani-lab/online_predatory_conversation_detection)")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def query(message):
    parsed_message = urllib.parse.quote(message)

    url = f"https://chatbot-api.vercel.app/api/?message=message={parsed_message}"

    response = requests.request("GET", url, headers=headers)
    return response.json()

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text 


user_input = get_text()

if user_input:
    output = query(user_input)

    st.session_state.past.append(user_input)
    print(output)

    st.session_state.generated.append(output["message"])

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')