from flask import Flask, render_template, jsonify, request
import json
from hmm_model.HMM import HMM
# crossdomain
from flask_cors import CORS, cross_origin


app = Flask(__name__, static_folder='web', template_folder='web')
CORS(app)


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/gen/<params>')
def gen(params):
    d = json.loads(params)
    h = HMM(d['A'], d['B'], d['states'], d['observations'])
    resp = h.gen_sequence(d['num'])
    return jsonify(data=resp)


@app.route('/viterbi/<params>')
def viterbi(params):
    d = json.loads(params)
    h = HMM(d['A'], d['B'], d['states'], d['observations'])
    resp = h.viterbi(d['seq'])
    return jsonify(data=resp)


@app.route('/train/<params>')
def train(params):
    d = json.loads(params)
    h = HMM(d['A'], d['B'], d['states'], d['observations'])
    h_trained = h.baum_welch(d['seq'])
    resp = {
        'A': h_trained.A.tolist(),
        'B': h_trained.B.tolist()
    }
    return jsonify(data=resp)

# @app.errorhandler(404)
# def page_not_found(error):
#     return render_template('page_not_found.html'), 404

if __name__ == "__main__":
    app.run()
