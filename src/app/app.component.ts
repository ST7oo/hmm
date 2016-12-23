/*
 * Angular 2 decorators and services
 */
import { Component, ViewEncapsulation } from '@angular/core';
import { Router } from '@angular/router';

import { AppState } from './app.service';

/*
 * App Component
 * Top Level Component
 */
@Component({
  selector: 'app',
  encapsulation: ViewEncapsulation.None,
  styleUrls: [
    './app.component.css'
  ],
  template: `
    <md-toolbar>
      <md-tab-group (focusChange)="changeTab($event)">
        <md-tab>
          <template md-tab-label>
            Home
          </template>
        </md-tab>
        <md-tab>
          <template md-tab-label>
            About
          </template>
        </md-tab>
      </md-tab-group>
    </md-toolbar>

    <main>
      <router-outlet></router-outlet>
    </main>
  `
})
export class AppComponent {

  constructor(
    public appState: AppState,
    private router: Router) {

  }

  ngOnInit() {
    console.log('Initial App State', this.appState.state);
  }

  changeTab(e) {
    switch (e.index) {
      case 0:
        this.router.navigateByUrl('/');
        break;
      case 1:
        this.router.navigateByUrl('/about');
        break;
    }
  }

}

/*
 * Please review the https://github.com/AngularClass/angular2-examples/ repo for
 * more angular app examples that you may copy/paste
 * (The examples may not be updated as quickly. Please open an issue on github for us to update it)
 * For help or questions please contact us at @AngularClass on twitter
 * or our chat on Slack at https://AngularClass.com/slack-join
 */
