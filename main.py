import model_ops
from flask import Flask, request, jsonify

model, tfidf = model_ops.load()
app = Flask(__name__)


@app.route('/', methods=['POST'])
def index():
    text = request.get_json()['text']
    res = model.predict(tfidf.transform([text]))
    print(text, "===", res)
    return jsonify({"positive": str(res[0])})


app.run()
