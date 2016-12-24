import { Injectable } from '@angular/core';

@Injectable()
export class HMMService {
  states: string[];
  observations: string[];
  A: number[][];
  B: number[][];

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
  }

  get_observation(i: number) {
    return this.observations[i.valueOf()];
  }

  insert_state(state: string) {
    this.states.push(state);
    this.A.push(new Array(this.states.length - 1).fill(0));
    this.B.push(new Array(this.observations.length).fill(0));
    for (let a of this.A) {
      a.push(0);
    }
  }

  insert_observation(observation: string) {
    this.observations.push(observation);
    for (let b of this.B) {
      b.push(0);
    }
  }

  delete_state(i: number) {
    this.states.splice(i.valueOf(), 1);
  }

  delete_observation(i: number) {
    this.observations.splice(i.valueOf(), 1);
  }

}
