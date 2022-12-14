from flask import Flask
from flask import request, jsonify
from flask_cors import CORS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Conversation, ConversationalPipeline

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
nlp = ConversationalPipeline(model=model, tokenizer=tokenizer)
conversation = Conversation()

app = Flask(__name__)
CORS(app)

@app.route('/add_input', methods = ['GET', 'POST'])
def add_input():
     text = request.json['text']
     conversation.add_user_input(text)
     result = nlp([conversation], do_sample=False, max_length=1000)
     messages = []
     for is_user, text in result.iter_texts():
          messages.append({
               'is_user': is_user,
               'text': text
          })
     print(messages)
     return jsonify({
         
          'messages': text
     })
@app.route('/reset', methods = ['GET', 'POST'])
def reset():
     global conversation
     conversation = Conversation()
     return 'ok' 

#give the chatbot an identity
@app.route('/init_persona', methods = ['GET', 'POST'])
def init():
     text = request.json['text']
     conversation.add_user_input('Hello')
     conversation.append_response(text)
     # Put the user's messages as "old message".
     conversation.mark_processed()
     return 'ok' 

if __name__ == "__main__":
    app.run(debug=True)