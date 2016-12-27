from flask import Flask, render_template, jsonify, redirect, url_for
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


@app.route('/gen/<params>')
def gen(params):
    d = json.loads(params)
    try:
        if len(d['states']) > 1 and len(d['observations']) > 1 and len(d['A']) > 1 and len(d['A'][0]) > 1 and len(d['B']) > 1 and len(d['B'][0]) > 1:
            h = HMM(d['A'], d['B'], d['states'], d['observations'])
            resp = jsonify(data=h.gen_sequence(d['num']))
        else:
            resp = jsonify(error='Incomplete model')
    except ValueError as e:
        resp = jsonify(error=str(e))
    return resp


@app.route('/viterbi/<params>')
def viterbi(params):
    d = json.loads(params)
    try:
        if len(d['states']) > 1 and len(d['observations']) > 1 and len(d['A']) > 1 and len(d['A'][0]) > 1 and len(d['B']) > 1 and len(d['B'][0]) > 1:
            h = HMM(d['A'], d['B'], d['states'], d['observations'])
            resp = jsonify(data=h.viterbi(d['seq']))
        else:
            resp = jsonify(error='Incomplete model')
    except ValueError as e:
        resp = jsonify(error=str(e))
    return resp


@app.route('/train/<params>')
def train(params):
    d = json.loads(params)
    try:
        if len(d['states']) > 1 and len(d['observations']) > 1 and len(d['A']) > 1 and len(d['A'][0]) > 1 and len(d['B']) > 1 and len(d['B'][0]) > 1:
            h = HMM(d['A'], d['B'], d['states'], d['observations'])
            max_iter = d['max_iter'] if 'max_iter' in d else 20
            h_trained, iters = h.baum_welch(d['seq'], max_iter)
            trained = {
                'A': h_trained.A.tolist(),
                'B': h_trained.B.tolist(),
                'iters': str(iters)
            }
            resp = jsonify(data=trained)
        else:
            resp = jsonify(error='Incomplete model')
    except ValueError as e:
        resp = jsonify(error=str(e))
    except KeyError as e:
        resp = jsonify(error=str(e))
    return resp

# @app.errorhandler(404)
# def page_not_found(error):
#     return render_template('page_not_found.html'), 404

if __name__ == "__main__":
    app.run()
