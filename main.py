import model_ops
from flask import Flask, request, jsonify

model, tfidf = model_ops.load()
app = Flask(__name__)


@app.route('/', methods=['POST'])
def index():
    text = request.get_json()['text']
    tonality = model.predict(tfidf.transform([text]))
    print(text, "===", tonality)
    return jsonify({"positive": str(tonality[0])})


app.run()
