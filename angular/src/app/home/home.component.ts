import { Component } from '@angular/core';

import { HMMService } from '../app.service';

@Component({
  selector: 'home',
  styleUrls: ['./home.component.css'],
  templateUrl: './home.component.html'
})
export class HomeComponent {
  generated_sequences = [[]];
  number_sequences = 1;
  sequence = ['C1', 'C2'];
  sequences = [['C1', 'C2', 'C3', 'C4', 'C4', 'C6', 'C7'],
  ['C2', 'C2', 'C5', 'C4', 'C4', 'C6', 'C6']];

  constructor(public hmm: HMMService) { }

  ngOnInit() {
    console.log('hello `Home` component');
  }

  insert_state(input: HTMLInputElement) {
    this.hmm.insert_state(input.value);
    input.value = null;
  }

  insert_observation(input: HTMLInputElement) {
    this.hmm.insert_observation(input.value);
    input.value = null;
  }

  delete_state(i) {
    this.hmm.delete_state(i);
  }

  delete_observation(i) {
    this.hmm.delete_observation(i);
  }

  add_obs(i) {
    this.sequence.push(this.hmm.get_observation(i));
  }

  remove_obs(i) {
    this.sequence.splice(i, 1);
  }

  add_sequence() {
    this.sequences.push(this.sequence);
    this.sequence = [];
  }

  remove_sequence(i) {
    this.sequences.splice(i, 1);
  }

  generate_sequence() {
    let gs = this.hmm.generate_sequence(this.number_sequences);
    gs.subscribe(r => this.generated_sequences = r);
  }

  viterbi() {

  }

  initialize_A(random: boolean) {
    this.hmm.initialize_matrix(true, false, random);
  }

  initialize_B(random: boolean) {
    this.hmm.initialize_matrix(false, true, random);
  }

  train() {
    console.log(this.hmm.A_ini);
    console.log(this.hmm.B_ini);
  }
}
