import { Injectable } from '@angular/core';

@Injectable()
export class HMMService {
  states: string[];
  observations: string[];
  A: number[][];
  B: number[][];
  A_ini: number[][];
  B_ini: number[][];

  constructor() {
    this.mock_data();
  }

  mock_data() {
    this.states = ['INIT', 'Onset', 'Mid', 'End', 'FINAL'];
    this.observations = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'];
    this.A = [[0, 1, 0, 0, 0],
    [0, 0.3, 0.7, 0, 0],
    [0, 0, 0.9, 0.1, 0],
    [0, 0, 0, 0.4, 0.6],
    [0, 0, 0, 0, 0]];
    this.B = [[0, 0, 0, 0, 0, 0, 0],
    [0.5, 0.2, 0.3, 0, 0, 0, 0],
    [0, 0, 0.2, 0.7, 0.1, 0, 0],
    [0, 0, 0, 0.1, 0, 0.5, 0.4],
    [0, 0, 0, 0, 0, 0, 0]];
    this.initialize_matrix(true, true);
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

  get_observation(i: number) {
    return this.observations[i.valueOf()];
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

}
