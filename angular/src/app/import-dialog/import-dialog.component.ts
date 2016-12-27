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
  text_sequences: string = '';
  message: string;
  param: string;

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
      } else if (variable == 'sequences') {
        this.text_sequences = result;
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
        this.text_sequences = '';
      } else if (format == '.emit') {
        this.text_emissions = data;
        this.text_transitions = '';
        this.text_sequences = '';
      } else if (format == '.input') {
        this.text_sequences = data;
        this.text_emissions = '';
        this.text_transitions = '';
      } else if (format == '.train') {
        this.text_sequences = data;
        this.text_emissions = '';
        this.text_transitions = '';
      }
    });
  }

  import() {
    this.message = '';
    let regexp = /\t|\s|,/;
    if (this.param == 'model') {
      let model = this.import_model(regexp);
      if (Object.keys(model).length > 0) {
        this.dialogRef.close(model);
      }
    } else if (this.param == 'sequences') {
      let sequences = this.import_sequences(regexp);
      if (sequences.length > 0) {
        this.dialogRef.close(sequences);
      }
    } else if (this.param == 'train') {
      let model = this.import_model(regexp);
      if (Object.keys(model).length > 0) {
        let sequences = this.import_sequences(regexp);
        if (sequences.length > 0) {
          model['train_seq'] = sequences;
          this.dialogRef.close(model);
        }
      }
    }
  }

  private import_model(regexp: RegExp): Object {
    if (this.text_transitions.length > 0 && this.text_emissions.length > 0) {
      let A = [];
      let B = [];
      let states = [];
      let observations = [];
      // Transitions
      let states_set = new Set();
      let lines = this.text_transitions.split('\n').filter(x => x.length > 0);
      // get states
      for (let line of lines) {
        let columns = line.split(regexp).filter(x => x.length > 0);
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
          let columns = line.split(regexp).filter(x => x.length > 0);
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
        lines = this.text_emissions.split('\n').filter(x => x.length > 0);
        // get observations
        for (let line of lines) {
          let columns = line.split(regexp).filter(x => x.length > 0);
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
            let columns = line.split(regexp).filter(x => x.length > 0);
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
        return model;
      } else {
        this.message = 'The model is incorrect. See the examples.'
      }
    } else {
      this.message = 'Transitions or Emissions cannot be empty';
    }
    return {};
  }

  private import_sequences(regexp: RegExp): Array<string[]> {
    if (this.text_sequences.length > 0) {
      let sequences = [];
      let lines = this.text_sequences.split('\n').filter(x => x.length > 0);
      for (let line of lines) {
        let columns = line.split(regexp).filter(x => x.length > 0);
        sequences.push(columns);
      }
      if (sequences.length > 0) {
        return sequences;
      }
      else {
        this.message = 'The file is incorrect. See the examples.'
      }
    } else {
      this.message = 'Select a file with sequences.';
    }
    return [];
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