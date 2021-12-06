import React, { Component } from "react";
import AuthService from "../services/auth.service";
import { Route } from "react-router-dom";

export default class Profile extends Component {
  constructor(props) {
    super(props);

    this.state = {
      redirect: null,
      userReady: false,
      currentUser: undefined,
    };
  }

  componentDidMount() {
    let currentUser = AuthService.getCurrentUser();

    if (!currentUser) this.setState({ redirect: "/login" });
    this.setState({ currentUser: currentUser, userReady: true });
    console.log(this.state.currentUser)
  }

  render() {
    if (!this.state.currentUser) {
      return <p>Loading...</p>
    }
    return (
      <Route
        path="/"
        component={() => {
          window.location.href = "http://127.0.0.1:5000/video?genres=" + this.state.currentUser.genres.join(',');
          return null;
        }}
      />
    );
  }
}
