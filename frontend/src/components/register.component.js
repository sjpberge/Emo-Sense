import React, { Component } from "react";
import Form from "react-validation/build/form";
import Input from "react-validation/build/input";
import CheckButton from "react-validation/build/button";
import { isEmail } from "validator";
import Multiselect from "multiselect-react-dropdown";

import AuthService from "../services/auth.service";

const required = (value) => {
  if (!value) {
    return (
      <div className="alert alert-danger" role="alert">
        This field is required!
      </div>
    );
  }
};

const email = (value) => {
  if (!isEmail(value)) {
    return (
      <div className="alert alert-danger" role="alert">
        This is not a valid email!
      </div>
    );
  }
};

const vusername = (value) => {
  if (value.length < 4 || value.length > 16) {
    return (
      <div className="alert alert-danger" role="alert">
        The username must be between 4 and 16 characters!
      </div>
    );
  }
};

const vpassword = (value) => {
  if (value.length < 8 || value.length > 40) {
    return (
      <div className="alert alert-danger" role="alert">
        The password must be between 8 and 40 characters!
      </div>
    );
  }
};

export default class Register extends Component {
  constructor(props) {
    super(props);
    this.handleRegister = this.handleRegister.bind(this);
    this.onChangeUsername = this.onChangeUsername.bind(this);
    this.onChangeEmail = this.onChangeEmail.bind(this);
    this.onChangePassword = this.onChangePassword.bind(this);
    this.onSelectGenre = this.onSelectGenre.bind(this);

    this.state = {
      username: "",
      email: "",
      password: "",
      successful: false,
      message: "",
      genres: [],
      selectedGenres: [],
      options: [
        {
          name: "Action",
          id: 0,
        },
        {
          name: "Adventure",
          id: 1,
        },
        {
          name: "Animation",
          id: 2,
        },
        {
          name: "Children's",
          id: 3,
        },
        {
          name: "Comedy",
          id: 4,
        },
        {
          name: "Crime",
          id: 5,
        },
        {
          name: "Documentary",
          id: 6,
        },
        {
          name: "Drama",
          id: 7,
        },
        {
          name: "Fantasy",
          id: 8,
        },
        {
          name: "Film-Noir",
          id: 9,
        },
        {
          name: "Horror",
          id: 10,
        },
        {
          name: "Musical",
          id: 11,
        },
        {
          name: "Mystery",
          id: 12,
        },
        {
          name: "Romance",
          id: 13,
        },
        {
          name: "Sci-Fi",
          id: 14,
        },
        {
          name: "Thriller",
          id: 15,
        },
        {
          name: "War",
          id: 16,
        },
        {
          name: "Western",
          id: 17,
        },
      ],
    };
  }

  onChangeUsername(e) {
    this.setState({
      username: e.target.value,
    });
  }

  onChangeEmail(e) {
    this.setState({
      email: e.target.value,
    });
  }

  onChangePassword(e) {
    this.setState({
      password: e.target.value,
    });
  }

  onSelectGenre(_, selectedItem) {
    this.setState((prevState) => ({
      genres: [...prevState.genres, selectedItem.id],
      selectedGenres: [...prevState.selectedGenres, selectedItem],
    }));
  }

  handleRegister(e) {
    e.preventDefault();

    this.setState({
      message: "",
      successful: false,
    });

    this.form.validateAll();
    console.log(this.state.genres);

    if (this.checkBtn.context._errors.length === 0) {
      AuthService.register(
        this.state.username,
        this.state.email,
        this.state.password,
        this.state.genres
      ).then(
        (response) => {
          this.setState({
            message: response.data.message,
            successful: true,
          });
        },
        (error) => {
          const resMessage =
            (error.response &&
              error.response.data &&
              error.response.data.message) ||
            error.message ||
            error.toString();

          this.setState({
            successful: false,
            message: resMessage,
          });
        }
      );
    }
  }

  render() {
    return (
      <div className="col-md-12">
        <div className="card card-container">
          <img
            src="//ssl.gstatic.com/accounts/ui/avatar_2x.png"
            alt="profile-img"
            className="profile-img-card"
          />

          <Form
            onSubmit={this.handleRegister}
            ref={(c) => {
              this.form = c;
            }}
          >
            {!this.state.successful && (
              <div>
                <div className="form-group">
                  <label htmlFor="username">Username</label>
                  <Input
                    type="text"
                    className="form-control"
                    name="username"
                    value={this.state.username}
                    onChange={this.onChangeUsername}
                    validations={[required, vusername]}
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="email">Email</label>
                  <Input
                    type="text"
                    className="form-control"
                    name="email"
                    value={this.state.email}
                    onChange={this.onChangeEmail}
                    validations={[required, email]}
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="password">Password</label>
                  <Input
                    type="password"
                    className="form-control"
                    name="password"
                    value={this.state.password}
                    onChange={this.onChangePassword}
                    validations={[required, vpassword]}
                  />
                </div>
                <div className="form-group">
                  <Multiselect
                    placeholder="Genres"
                    options={this.state.options} // Options to display in the dropdown
                    selectedValues={this.state.selectedGenres} // Preselected value to persist in dropdown
                    onSelect={this.onSelectGenre} // Function will trigger on select event
                    onRemove={this.onRemove} // Function will trigger on remove event
                    displayValue="name" // Property name to display in the dropdown options
                  />
                </div>
                <div className="form-group">
                  <button className="btn btn-primary btn-block">Sign Up</button>
                </div>
              </div>
            )}

            {this.state.message && (
              <div className="form-group">
                <div
                  className={
                    this.state.successful
                      ? "alert alert-success"
                      : "alert alert-danger"
                  }
                  role="alert"
                >
                  {this.state.message}
                </div>
              </div>
            )}
            <CheckButton
              style={{ display: "none" }}
              ref={(c) => {
                this.checkBtn = c;
              }}
            />
          </Form>
        </div>
      </div>
    );
  }
}
