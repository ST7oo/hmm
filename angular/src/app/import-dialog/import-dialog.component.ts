import { Component } from '@angular/core';
import { Http } from '@angular/http';
import { MdDialogRef } from '@angular/material';

@Component({
  selector: 'import-dialog',
  styleUrls: ['import-dialog.component.css'],
  templateUrl: 'import-dialog.component.html'
})
export class ImportDialog {
  text_transitions: string = '';
  text_emissions: string = '';
  message: string;

  constructor(public dialogRef: MdDialogRef<ImportDialog>, private http: Http) { }

  file_change(event, variable) {
    this.read_file(event.target, variable);
  }

  read_file(input: any, variable: string) {
    let file: File = input.files[0];
    let reader: FileReader = new FileReader();
    reader.onloadend = (e) => {
      let result = reader.result.trim();
      if (variable == 'transitions') {
        this.text_transitions = result;
      } else if (variable == 'emissions') {
        this.text_emissions = result;
      }
    };
    reader.readAsText(file);
  }

  example(format: string) {
    this.http.get('assets/data/example' + format).subscribe(res => {
      let data = res.text().trim();
      if (format == '.trans') {
        this.text_transitions = data;
        this.text_emissions = '';
      } else if (format == '.emit') {
        this.text_emissions = data;
        this.text_transitions = '';
      }
    });
  }

  import() {
    this.message = '';
    if (this.text_transitions.length > 0 && this.text_emissions.length > 0) {
      let A = [];
      let B = [];
      let states = [];
      let observations = [];
      // Transitions
      let states_set = new Set();
      let lines = this.text_transitions.split('\n');
      // get states
      for (let line of lines) {
        let columns = line.split(/\t|\s\s\s\s/);
        if (columns.length == 3) {
          states_set.add(columns[0]);
          states_set.add(columns[1]);
        }
      }
      states = Array.from(states_set);
      let N = states.length;
      if (N > 1) {
        for (let i = 0; i < N; i++) {
          A.push(new Array(N).fill(0));
        }
        // get probabilities
        for (let line of lines) {
          let columns = line.split(/\t|\s\s\s\s/);
          if (columns.length == 3) {
            if (this.is_probability(columns[2])) {
              A[states.indexOf(columns[0])][states.indexOf(columns[1])] = parseFloat(columns[2]);
            } else {
              this.message = 'The third column should be a probability';
            }
          }
        }
        // Emissions
        let observations_set = new Set();
        lines = this.text_emissions.split('\n');
        // get observations
        for (let line of lines) {
          let columns = line.split('\t');
          if (columns.length == 3) {
            observations_set.add(columns[1]);
          }
        }
        observations = Array.from(observations_set);
        let M = observations.length;
        if (M > 0) {
          for (let i = 0; i < N; i++) {
            B.push(new Array(M).fill(0));
          }
          // get probabilities
          for (let line of lines) {
            let columns = line.split('\t');
            if (columns.length == 3) {
              if (this.is_probability(columns[2])) {
                B[states.indexOf(columns[0])][observations.indexOf(columns[1])] = parseFloat(columns[2]);
              } else {
                this.message = 'The third column should be a probability';
              }
            }
          }
        } else {
          this.message = 'At least 1 observation must be in the model';
        }
      } else {
        this.message = 'At least 2 states must be in the model';
      }
      if (this.check_probabilities(A) && this.check_probabilities(B)) {
        let model = {
          states: states,
          observations: observations,
          A: A,
          B: B
        };
        this.dialogRef.close(model);
      } else {
        this.message = 'The model is incorrect. See the examples.'
      }
    } else {
      this.message = 'Transitions or Emissions cannot be empty';
    }
  }

  private is_probability(num: string) {
    let n = parseFloat(num);
    return !isNaN(n) && n >= 0 && n <= 1;
  }

  private check_probabilities(matrix: number[][]) {
    let correct = false;
    for (let row of matrix) {
      if (!row.every(x => x == 0)) {
        correct = true;
        break;
      }
    }
    return correct;
  }
}