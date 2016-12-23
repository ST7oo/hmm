import { Component } from '@angular/core';

@Component({
  selector: 'about',
  template: `
    <h1>About</h1>
    <p>Hidden Markov Model implemented in python</p>
    <p><i>Rodney Ledesma</i></p>
  `
})
export class AboutComponent {
  ngOnInit() {
    console.log('hello `About` component');
  }

}
