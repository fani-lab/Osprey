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
CORS(app)

classify_model = joblib.load('model.joblib') 
sentence_model = SentenceTransformer('average_word_embeddings_glove.6B.300d')

@app.route('/add_input', methods = ['GET', 'POST'])
def add_input():
     """_summary_

     Returns:
         json: Chatbot response
     """
     text = request.json['text']
     conversation.add_user_input(text) # Add user input to conversation
     result = nlp([conversation], do_sample=False, max_length=1000) # Get chatbot response
     messages = []

     #Browse results and form a list of messages
     for is_user, text in result.iter_texts():
          messages.append({
               'is_user': is_user,
               'text': text
          })
     sentence_embeddings = sentence_model.encode(testinput).reshape(1, -1) #map sentence to vector
     return jsonify({
          'messages': text
     })
     
@app.route('/reset', methods = ['GET', 'POST'])
def reset():
     """Reset conversation, so that the chatbot forgets everything."""
     global conversation
     conversation = Conversation()

@app.route('/init_persona', methods = ['GET', 'POST'])
def init():
     """give the chatbot an identity"""
     text = request.json['text']
     conversation.add_user_input('Hello') # User doesn't need to say hello at the start
     conversation.append_response(text) # Personality of the chatbot
     conversation.mark_processed() # Archive the previous messages and consider them as a context 

if __name__ == "__main__":
    app.run(debug=True)