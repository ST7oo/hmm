import { Injectable } from '@angular/core';
import { Http, Response, Headers } from '@angular/http';
import { Observable } from 'rxjs/Rx';

@Injectable()
export class HMMService {
  states: string[];
  observations: string[];
  A: number[][];
  B: number[][];
  A_ini: number[][];
  B_ini: number[][];
  sequences: string[][];
  path: string[][];
  path_probabilities: number[];
  train_seq: string[][];
  host: string;
  headers: Headers;
  // env = 'dev';
  env = 'prod';

  constructor(private http: Http) {
    this.mock_data();
    this.headers = new Headers();
    this.headers.append('Content-Type', 'application/json');
    if (this.env == 'prod') {
      this.host = '/';
    } else {
      this.host = 'http://localhost:5000/';
    }
  }

  mock_data() {
    this.http.get('assets/data/mock_data.json').subscribe(res => {
      let data = res.json();
      this.states = data.states;
      this.observations = data.observations;
      this.A = data.A;
      this.B = data.B;
      this.A_ini = data.A_ini;
      this.B_ini = data.B_ini;
      this.sequences = data.sequences;
      this.train_seq = data.train_seq;
      this.initialize_path();
    });
  }

  reset(param: string) {
    if (param == 'model') {
      this.states = [];
      this.observations = [];
      this.A = [[]];
      this.B = [[]];
    } else if (param == 'sequences') {
      this.sequences = [];
    } else if (param == 'train') {
      this.states = [];
      this.observations = [];
      this.A_ini = [[]];
      this.B_ini = [[]];
      this.train_seq = [];
    }
  }

  initialize_matrix(a: boolean, b: boolean, random?: boolean) {
    let N = this.states.length;
    let M = this.observations.length;
    let A = [];
    let B = [];
    if (random) {
      for (let i = 0; i < N; i++) {
        if (a) {
          let tmp = Array.from({ length: N }, () => Math.random());
          let sum = tmp.reduce((i, j) => i + j, 0);
          A.push(tmp.map(v => v / sum));
        }
        if (b) {
          let tmp = Array.from({ length: M }, () => Math.random());
          let sum = tmp.reduce((i, j) => i + j, 0);
          B.push(tmp.map(v => v / sum));
        }
      }
    }
    else {
      for (let i = 0; i < N; i++) {
        if (a) {
          A.push(new Array(N).fill(1 / N));
        }
        if (b) {
          B.push(new Array(M).fill(1 / M));
        }
      }
    }
    if (a) {
      this.A_ini = A;
    }
    if (b) {
      this.B_ini = B;
    }
  }

  initialize_path() {
    this.path = [];
    this.path_probabilities = [];
    for (let s of this.sequences) {
      this.path.push(new Array(s.length).fill('...'));
      this.path_probabilities.push(0);
    }
  }

  get_state(i: number) {
    return this.states[i];
  }

  get_observation(i: number) {
    return this.observations[i];
  }

  set_states(states: string[]) {
    this.states = states;
  }

  set_observations(observations: string[]) {
    this.observations = observations;
  }

  set_sequences(sequences: string[][]) {
    this.sequences = sequences;
    this.initialize_path();
  }

  set_train_seq(sequences: string[][]) {
    this.train_seq = sequences;
  }

  set_A(A: number[][]) {
    this.A = A;
  }

  set_B(B: number[][]) {
    this.B = B;
  }

  set_A_ini(A: number[][]) {
    this.A_ini = A;
  }

  set_B_ini(B: number[][]) {
    this.B_ini = B;
  }

  insert_state(state: string) {
    this.states.push(state);
    this.A.push(new Array(this.states.length - 1).fill(0));
    this.A_ini.push(new Array(this.states.length - 1).fill(0));
    this.B.push(new Array(this.observations.length).fill(0));
    this.B_ini.push(new Array(this.observations.length).fill(0));
    for (let i in this.A) {
      this.A[i].push(0);
      this.A_ini[i].push(0);
    }
  }

  insert_observation(observation: string) {
    this.observations.push(observation);
    for (let i in this.B) {
      this.B[i].push(0);
      this.B_ini[i].push(0);
    }
  }

  delete_state(i: number) {
    this.states.splice(i.valueOf(), 1);
  }

  delete_observation(i: number) {
    this.observations.splice(i.valueOf(), 1);
  }

  insert_sequence(seq: string[]) {
    this.sequences.push(seq);
    this.path.push(new Array(seq.length).fill('...'));
  }

  delete_sequence(i: number) {
    this.sequences.splice(i, 1);
    this.path.splice(i, 1);
  }

  generate_sequence(num: number) {
    let data = this.prepare_model();
    data['num'] = num;
    let json_data = JSON.stringify(data);
    if (this.env == 'prod') {
      return this.http.post(this.host + 'gen', json_data, { headers: this.headers }).map(this.handle_response).catch(this.handle_error);
    } else {
      return this.http.get(this.host + 'gen?data=' + json_data).map(this.handle_response).catch(this.handle_error);
    }
  }

  viterbi() {
    let data = this.prepare_model();
    data['seq'] = this.sequences;
    let json_data = JSON.stringify(data);
    if (this.env == 'prod') {
      return this.http.post(this.host + 'viterbi', json_data, { headers: this.headers }).map(res => {
        let r = res.json();
        if (r.data) {
          let path = [];
          let probabilities = [];
          for (let s of r.data) {
            path.push(s[0]);
            probabilities.push(s[1]);
          }
          this.path = path;
          this.path_probabilities = probabilities;
          return { data: true };
        } else {
          return r;
        }
      }).catch(this.handle_error);
    } else {
      return this.http.get(this.host + 'viterbi?data=' + json_data).map(res => {
        let r = res.json();
        if (r.data) {
          let path = [];
          let probabilities = [];
          for (let s of r.data) {
            path.push(s[0]);
            probabilities.push(s[1]);
          }
          this.path = path;
          this.path_probabilities = probabilities;
          return { data: true };
        } else {
          return r;
        }
      }).catch(this.handle_error);
    }
  }

  train(max_iter: number) {
    let data = this.prepare_model(true);
    data['seq'] = this.train_seq;
    data['max_iter'] = max_iter;
    let json_data = JSON.stringify(data);
    if (this.env == 'prod') {
      return this.http.post(this.host + 'train', json_data, { headers: this.headers }).map(this.handle_response).catch(this.handle_error);
    } else {
      return this.http.get(this.host + 'train?data=' + json_data).map(this.handle_response).catch(this.handle_error);
    }
  }

  private prepare_model(ini?: boolean) {
    let A = [];
    let B = [];
    let N = this.states.length;
    let M = this.observations.length;
    if (!ini) {
      for (let i = 0; i < N; i++) {
        A.push(this.A[i].slice(0, N))
        B.push(this.B[i].slice(0, M));
      }
    }
    else {
      for (let i = 0; i < N; i++) {
        A.push(this.A_ini[i].slice(0, N))
        B.push(this.B_ini[i].slice(0, M));
      }
    }
    let model = {
      A: A,
      B: B,
      states: this.states,
      observations: this.observations
    };
    return model;
  }

  private handle_response(res: Response) {
    let ret;
    try {
      ret = res.json();
    } catch (error) {
      ret = { error: 'Unexpected error' }
    }
    return ret;
  }

  private handle_error(error: Response | any) {
    let errMsg: string;
    if (error instanceof Response) {
      const body = error.json() || '';
      const err = body.error || JSON.stringify(body);
      errMsg = `${error.status} - ${error.statusText || ''} ${err}`;
    } else {
      errMsg = error.message ? error.message : error.toString();
    }
    // console.error(errMsg);
    return Observable.throw(errMsg);
  }

}
