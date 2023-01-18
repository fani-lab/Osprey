from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Conversation, ConversationalPipeline
from sentence_transformers import SentenceTransformer
import joblib
from scipy import sparse
import datetime

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
nlp = ConversationalPipeline(model=model, tokenizer=tokenizer)
conversation = Conversation()

app = Flask(__name__)
CORS(app) # Allow cross-origin requests

classify_model = joblib.load('bigboi.joblib') 
sentence_model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
messages = []

@app.route('/add_input', methods = ['GET', 'POST'])
def add_input():
     """Receieve chatbot response and whenever or not it's predatorial

     Returns:
         json: Chatbot response
     """
     userinput = request.json['text']
     time = request.json['time']  
     conversation.add_user_input(userinput) # Add user input to conversation
     result = nlp([conversation], do_sample=False, max_length=1000) # Get chatbot response
     global messages
     #Browse results and form a list of messages
     for is_user, chatbot_response in result.iter_texts():
          messages.append({
               'is_user': is_user,
               'text': chatbot_response
          })
     print(messages)
     #print(chatbot_response, len(chatbot_response.split(' ')), len(chatbot_response))

     chatbot_label = classify_msg(chatbot_response, float(time)) #len(chatbot_response.split(' ')), len(chatbot_response), int(hour) )
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
     usertime = request.json['time']
     print('usertime', float(usertime))
     user_label = classify_msg(userinput, float(usertime) ) #len(userinput.split(" ")), len(userinput), 1672620360)
     return jsonify({
          "user_label": str(user_label),
     })

@app.route('/reset', methods = ['GET', 'POST'])
def reset():
     """Reset conversation, so that the chatbot forgets everything."""
     global conversation
     global messages
     messages = []
     conversation = Conversation()
     return "OK"

@app.route('/init_persona', methods = ['GET', 'POST'])
def init():
     """Give the chatbot an identity"""
     text = request.json['text']
     conversation.add_user_input('Hello') # User doesn't need to say hello at the start
     conversation.append_response(text) # Personality of the chatbot
     conversation.mark_processed() # Archive the previous messages and consider them as a context 
     return "OK"

def classify_msg(msg, time=00.00):
     """Embed the message and classify it using the classification model

     Args:
         msg (str): Message input.

     Returns:
         list[int]: returns 1 or 0 for predatorial or not predatorial respectively.
     """
     global messages #previous messages
     msg_line=[len(messages)]; 
     char_count=[len(msg)]; 
     word_count=[len(msg.split(' '))]; 
     nauthor=[2]

     prv_cat =""
     if len(messages) > 0:
          for i in messages:
               prv_cat += " "+i['text']

     prv_msg = sentence_model.encode(prv_cat).reshape(1, -1)

     encoded_msg = sentence_model.encode(msg).reshape(1, -1)
     features = sparse.csr_matrix((0,  1)).transpose()

     features = sparse.csr_matrix(sparse.hstack((
     features,
     encoded_msg, 
     #char_count,
     word_count,
     prv_msg,
     nauthor,
     time,
     msg_line
     )))

     label = classify_model.predict(features)
     print(label)
     return label[0]

if __name__ == "__main__":
    app.run(debug=True)