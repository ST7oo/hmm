import { Component } from '@angular/core';

@Component({
  selector: 'home',
  styleUrls: ['./home.component.css'],
  templateUrl: './home.component.html'
})
export class HomeComponent {

  ngOnInit() {
    console.log('hello `Home` component');
  }
}
