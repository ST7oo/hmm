from flask import Flask, render_template, jsonify, redirect, url_for, request
import json
from hmm_model.HMM import HMM
# crossdomain
from flask_cors import CORS, cross_origin


app = Flask(__name__, static_folder='web', template_folder='web')
CORS(app)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/web/")
def web():
    return redirect(url_for('index'))


@app.route('/gen', methods=['GET', 'POST', 'OPTIONS'])
def gen():
    if request.method == 'GET':
        d = json.loads(request.args.getlist('data')[0])
    else:
        d = request.get_json()
    try:
        if len(d['states']) > 1 and len(d['observations']) > 1 and len(d['A']) > 1 and len(d['A'][0]) > 1 and len(d['B']) > 1 and len(d['B'][0]) > 1:
            h = HMM(d['states'], d['observations'], d['A'], d['B'])
            resp = jsonify(data=h.gen_sequence(d['num']))
        else:
            resp = jsonify(error='Incomplete model')
    except ValueError as e:
        resp = jsonify(error=str(e))
    return resp


@app.route('/viterbi', methods=['GET', 'POST', 'OPTIONS'])
def viterbi():
    if request.method == 'GET':
        d = json.loads(request.args.getlist('data')[0])
    else:
        d = request.get_json()
    try:
        if len(d['states']) > 1 and len(d['observations']) > 1 and len(d['A']) > 1 and len(d['A'][0]) > 1 and len(d['B']) > 1 and len(d['B'][0]) > 1:
            h = HMM(d['states'], d['observations'], d['A'], d['B'])
            resp = jsonify(data=h.viterbi(d['seq']))
        else:
            resp = jsonify(error='Incomplete model')
    except ValueError as e:
        resp = jsonify(error=str(e))
    return resp


@app.route('/train', methods=['GET', 'POST', 'OPTIONS'])
def train():
    if request.method == 'GET':
        d = json.loads(request.args.getlist('data')[0])
    else:
        d = request.get_json()
    try:
        if len(d['states']) > 1 and len(d['observations']) > 1 and len(d['A']) > 1 and len(d['A'][0]) > 1 and len(d['B']) > 1 and len(d['B'][0]) > 1:
            h = HMM(d['states'], d['observations'], d['A'], d['B'])
            max_iter = d['max_iter'] if 'max_iter' in d else 20
            iterations = h.baum_welch(d['seq'], max_iter)
            trained = {
                'A': h.A.tolist(),
                'B': h.B.tolist(),
                'iterations': str(iterations)
            }
            resp = jsonify(data=trained)
        else:
            resp = jsonify(error='Incomplete model')
    except ValueError as e:
        resp = jsonify(error=str(e))
    except KeyError as e:
        resp = jsonify(error=str(e))
    return resp


if __name__ == "__main__":
    app.run()
