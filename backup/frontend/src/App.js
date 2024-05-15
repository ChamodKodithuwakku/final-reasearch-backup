import './App.css';
import Homepage from "./components/homepage/Homepage";
import Login from "./components/login/Login";
import Register from "./components/register/Register";
import Playlist from "./components/PlaylistPage/PlaylistPage";
import NavBar from "./components/NavBar/NavBar";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import { useState } from 'react';

function App() {
    const [user, setLoginUser] = useState({});
    const [recognizedEmotion, setRecognizedEmotion] = useState('');

    return (
        <div className="App">
            <Router>
                <NavBar user={user} />
                <Switch>
                    <Route exact path="/">
                        {user && user._id ? <Homepage user={user}/> : <Login setLoginUser={setLoginUser} />}
                    </Route>
                    <Route path="/Login">
                        <Login setLoginUser={setLoginUser} />
                    </Route>
                    <Route path="/Register">
                        <Register />
                    </Route>
                    <Route path="/playlist/:emotion" component={Playlist} />
                </Switch>
            </Router>
        </div>
    );
}

export default App;
