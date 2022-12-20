from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Conversation, ConversationalPipeline
from sentence_transformers import SentenceTransformer
import joblib

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
nlp = ConversationalPipeline(model=model, tokenizer=tokenizer)
conversation = Conversation()

app = Flask(__name__)
CORS(app) # Allow cross-origin requests

classify_model = joblib.load('model.joblib') 
sentence_model = SentenceTransformer('average_word_embeddings_glove.6B.300d')

@app.route('/add_input', methods = ['GET', 'POST'])
def add_input():
     """Receieve chatbot response and whenever or not it's predatorial

     Returns:
         json: Chatbot response
     """
     userinput = request.json['text']
     conversation.add_user_input(userinput) # Add user input to conversation
     result = nlp([conversation], do_sample=False, max_length=1000) # Get chatbot response
     messages = []

     #Browse results and form a list of messages
     for is_user, chatbot_response in result.iter_texts():
          messages.append({
               'is_user': is_user,
               'text': chatbot_response
          })
     print(chatbot_response)

     chatbot_label = classify_msg(chatbot_response)
     return jsonify({
          'messages': chatbot_response,
          "chatbot_label": str(chatbot_label),
     })

@app.route('/classify_user_msg', methods = ['GET', 'POST'])
def classify_user_msg():
     """Check if user's message is predatorial or not

     Returns:
         json: Classification model's prediction
     """
     userinput = request.json['text']
     user_label = classify_msg(userinput)
     return jsonify({
          "user_label": str(user_label),
     })

@app.route('/reset', methods = ['GET', 'POST'])
def reset():
     """Reset conversation, so that the chatbot forgets everything."""
     global conversation
     conversation = Conversation()

@app.route('/init_persona', methods = ['GET', 'POST'])
def init():
     """Give the chatbot an identity"""
     text = request.json['text']
     conversation.add_user_input('Hello') # User doesn't need to say hello at the start
     conversation.append_response(text) # Personality of the chatbot
     conversation.mark_processed() # Archive the previous messages and consider them as a context 
     
def classify_msg(msg):
     """Embed the message and classify it using the classification model

     Args:
         msg (str): Message input.

     Returns:
         list[int]: returns 1 or 0 for predatorial or not predatorial respectively.
     """
     encoded_msg = sentence_model.encode(msg).reshape(1, -1)
     label = classify_model.predict(encoded_msg)
     return label[0]

if __name__ == "__main__":
    app.run(debug=True)