# Hidden Markov Model Interactive

Hidden Markov Model implemented in Python 3.5 with a web interface using Flask (back-end) and Angular 2 (front-end).

> [Live demo](https://hmm-interactive.herokuapp.com/)

The logic of the Hidden Markov Model is in `hmm_model/HMM.py`.

### Development

 - Install Python
 - Install Node.js
 - (Optionally) Install Yarn (otherwise replace `yarn` for `npm`)
 - Clone repository
 - Install dependencies
	 - `pip install -r requirements.txt`
	 - `cd angular`
	 - `yarn install`
 - Change the files:
	 - `angular/config/webpack.common.js` line 29 should be `baseUrl: '/'`
	 - `angular/src/app/app.service.ts` line 19 should be `env = 'dev'`
 - Run the frontend server (in the `angular` folder)
	 - `yarn start`
 - In another terminal run the backend server in debug mode
	 - `cd ..`
	 - `export FLASK_APP=app.py`
	 - `export FLASK_DEBUG=1`
	 - `flask run`
 - The app runs at http://localhost:3000/

### Deployment
 - Change the files:
	 - `angular/config/webpack.common.js` line 29 should be `baseUrl: '/web/'`
	 - `angular/src/app/app.service.ts` line 19 should be `env = 'prod'`
 - Build with `yarn build:prod`
 - Copy the contents of the folder `angular/dist` to `web`
 - If the backend server is running go to http://localhost:5000/