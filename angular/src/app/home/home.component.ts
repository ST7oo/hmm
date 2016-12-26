import { Component } from '@angular/core';
import { MdDialog, MdDialogRef } from '@angular/material';

import { HMMService } from '../app.service';
import { ImportDialog } from '../import-dialog';

@Component({
  selector: 'home',
  styleUrls: ['./home.component.css'],
  templateUrl: './home.component.html'
})
export class HomeComponent {
  iters: number;
  error_generate: string;
  error_viterbi: string;
  error_train: string;
  A_trained: string[][];
  B_trained: string[][];
  generated_sequences = [[]];
  number_sequences = 1;
  max_iter = 10;
  sequence = ['C1', 'C2'];
  calculating_train = false;
  trained = false;
  dialogRef: MdDialogRef<ImportDialog>;

  constructor(public hmm: HMMService, public dialog: MdDialog) { }

  ngOnInit() {
    console.log('Home component');
  }

  import() {
    this.dialogRef = this.dialog.open(ImportDialog, {
      disableClose: false
    });
    this.dialogRef.afterClosed().subscribe(res => {
      this.dialogRef = null;
      if (res) {
        this.hmm.set_states(res.states);
        this.hmm.set_observations(res.observations);
        this.hmm.set_A(res.A);
        this.hmm.set_B(res.B);
      }
    })
  }

  add_state(input: HTMLInputElement) {
    this.hmm.insert_state(input.value);
    input.value = null;
  }

  add_observation(input: HTMLInputElement) {
    this.hmm.insert_observation(input.value);
    input.value = null;
  }

  remove_state(i) {
    this.hmm.delete_state(i);
  }

  remove_observation(i) {
    this.hmm.delete_observation(i);
  }

  add_obs(i) {
    this.sequence.push(this.hmm.get_observation(i));
  }

  remove_obs(i) {
    this.sequence.splice(i, 1);
  }

  add_sequence() {
    if (this.sequence.length > 0) {
      this.hmm.insert_sequence(this.sequence);
      this.sequence = [];
    }
  }

  remove_sequence(i) {
    this.hmm.delete_sequence(i);
  }

  initialize_A(random: boolean) {
    this.hmm.initialize_matrix(true, false, random);
  }

  initialize_B(random: boolean) {
    this.hmm.initialize_matrix(false, true, random);
  }

  generate_sequence() {
    this.error_generate = '';
    let gs = this.hmm.generate_sequence(this.number_sequences);
    gs.subscribe(r => {
      if (r.data) {
        this.generated_sequences = r.data;
      } else if (r.error) {
        this.error_generate = r.error;
      } else {
        this.error_generate = 'Unexpected error ocurred.';
      }
    });
  }

  viterbi() {
    this.error_viterbi = '';
    let v = this.hmm.viterbi();
    let paths = [];
    let probabilities = [];
    v.subscribe(r => {
      if (!r.data) {
        if (r.error) {
          this.error_viterbi = r.error;
        } else {
          this.error_viterbi = 'Unexpected error ocurred.';
        }
      }
    });
  }

  train() {
    this.calculating_train = true;
    let bw = this.hmm.train(this.max_iter);
    bw.subscribe(r => {
      if (r.data) {
        this.A_trained = r.data.A;
        this.B_trained = r.data.B;
        this.iters = r.data.iters;
        this.trained = true;
      } else if (r.error) {
        this.error_train = r.error;
        this.trained = false;
      } else {
        this.error_train = 'Unexpected error ocurred.';
        this.trained = false;
      }
      this.calculating_train = false;
    });
  }
}
