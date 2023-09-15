from project.signup_and_login.forms import (
    RegistrationForm,
    LoginForm,
    UpdateAccountForm,
)


def add_admin_user_and_login(client):
    # add a admin user
    form = RegistrationForm()
    form.username.data = "admin"
    form.email.data = "admin@example.com"
    form.password.data = "password"
    form.confirm_password.data = "password"
    response = client.post("/register", data=form.data, follow_redirects=True)

    # login with admin
    form = LoginForm()
    form.email.data = "admin@example.com"
    form.password.data = "password"
    assert form.validate() is True
    response = client.post("/login", data=form.data, follow_redirects=True)
    assert b"Login Successful" in response.data


def add_user_and_login(client, name="test"):
    # add a admin user
    form = RegistrationForm()
    form.username.data = f"{name}"
    form.email.data = f"{name}@example.com"
    form.password.data = "password"
    form.confirm_password.data = "password"
    response = client.post("/register", data=form.data, follow_redirects=True)

    # login with admin
    form = LoginForm()
    form.email.data = f"{name}@example.com"
    form.password.data = "password"
    assert form.validate() is True
    response = client.post("/login", data=form.data, follow_redirects=True)
    assert b"Login Successful" in response.data
