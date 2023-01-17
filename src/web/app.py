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

classify_model = joblib.load('model.joblib') 
#classify_model = joblib.load('model.joblib')
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
     print(messages)
     #print(chatbot_response, len(chatbot_response.split(' ')), len(chatbot_response))

     hour = datetime.datetime.now().strftime("%H")
     minute = datetime.datetime.now().strftime("%M")
     #time = datetime(2023, 1,1,int(hour), int(minute)).timestamp()
     print('hour',hour)
     print('minute',minute)
     chatbot_label = classify_msg(chatbot_response) #len(chatbot_response.split(' ')), len(chatbot_response), int(hour) )
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
     print('usertime',usertime)
     user_label = classify_msg(userinput,) #len(userinput.split(" ")), len(userinput), 1672620360)
     return jsonify({
          "user_label": str(user_label),
     })

@app.route('/reset', methods = ['GET', 'POST'])
def reset():
     """Reset conversation, so that the chatbot forgets everything."""
     global conversation
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

def classify_msg(msg):
     """Embed the message and classify it using the classification model

     Args:
         msg (str): Message input.

     Returns:
         list[int]: returns 1 or 0 for predatorial or not predatorial respectively.
     """
     encoded_msg = sentence_model.encode(msg).reshape(1, -1)
     #features = sparse.csr_matrix(sparse.hstack((1,
     #encoded_msg, 
     #)))
     #label = classify_model.predict(features)
     label = classify_model.predict(encoded_msg)
     #label = classify_model.predict(sparse.csr_matrix(sparse.hstack((encoded_msg, 1, 3, 1672620360))))
     print(label)
     return label[0]

if __name__ == "__main__":
    app.run(debug=True)