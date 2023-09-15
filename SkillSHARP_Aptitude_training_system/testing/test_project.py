from project.models import User, Questions
from project.signup_and_login.forms import (
    RegistrationForm,
    LoginForm,
    ResetPasswordForm,
    UpdateAccountForm,
)
from project.admin.forms import AddQuestionForm, UpdateQuestionForm
from project import bcrypt
from testing.utils import add_admin_user_and_login, add_user_and_login


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"HOME" in response.data


def test_registration_page(client, app):
    with app.app_context():
        # test get request
        response = client.get("/register")
        assert response.status_code == 200

        # testing form
        form = RegistrationForm()
        assert form.validate() is False

        # test post request with invalid data
        form.username.data = "testuser"
        form.email.data = "testuser@example.com"
        form.password.data = "pass"
        form.confirm_password.data = "pass"
        assert form.validate() is False
        response = client.post("/register", data=form.data, follow_redirects=True)
        assert (
            b"Registration Unsuccessful. Please check your details and try again"
            in response.data
        )

        # test post request with valid data
        form.username.data = "testuser"
        form.email.data = "testuser@example.com"
        form.password.data = "password"
        form.confirm_password.data = "password"
        assert form.validate() is True
        response = client.post("/register", data=form.data, follow_redirects=True)
        assert (
            b"Your account has been created! You are now able to log in"
            in response.data
        )

        # test if user is created
        user = User.query.filter_by(username="testuser").first()
        assert user is not None
        assert user.email == "testuser@example.com"
        assert user.username == "testuser"
        assert User.query.count() == 1

        # test if we can use same username to register
        form.username.data = "testuser"
        form.email.data = "test@example.com"
        form.password.data = "password"
        form.confirm_password.data = "password"
        assert form.validate() is False

        # test if we can use same email to register
        form.username.data = "testuser2"
        form.email.data = "testuser@example.com"
        form.password.data = "password"
        form.confirm_password.data = "password"
        assert form.validate() is False

        # test if we can add new user
        form.username.data = "testuser2"
        form.email.data = "testuser2@example.com"
        form.password.data = "password"
        form.confirm_password.data = "password"
        assert form.validate() is True
        response = client.post("/register", data=form.data, follow_redirects=True)
        assert (
            b"Your account has been created! You are now able to log in"
            in response.data
        )
        assert User.query.count() == 2

        # test if password matches
        user = User.query.filter_by(username=form.username.data).first()
        assert bcrypt.check_password_hash(user.password, form.password.data) is True


def test_login_page(client, app):
    with app.app_context():
        # test get request
        response = client.get("/login")
        assert response.status_code == 200

        # add a user
        form = RegistrationForm()
        form.username.data = "testuser"
        form.email.data = "testuser@example.com"
        form.password.data = "password"
        form.confirm_password.data = "password"
        response = client.post("/register", data=form.data, follow_redirects=True)

        # login with invalid user
        form = LoginForm()
        form.email.data = "test@example.com"
        form.password.data = "password"
        assert form.validate() is True
        response = client.post("/login", data=form.data, follow_redirects=True)
        assert b"Login Unsuccessful. User does not exist" in response.data

        # login with invalid password
        form = LoginForm()
        form.email.data = "testuser@example.com"
        form.password.data = "wrongpassword"
        assert form.validate() is True
        response = client.post("/login", data=form.data, follow_redirects=True)
        assert b"Login Unsuccessful. Wrong password" in response.data

        # login with valid data
        form = LoginForm()
        form.email.data = "testuser@example.com"
        form.password.data = "password"
        assert form.validate() is True
        response = client.post("/login", data=form.data, follow_redirects=True)
        assert b"Login Successful" in response.data


def test_account_page(client, app):
    with app.app_context():
        # test get request without login
        response = client.get("/account", follow_redirects=True)
        assert b"Please log in to access this page" in response.data

        # test get request with login
        add_user_and_login(client)
        response = client.get("/account")
        assert b"<title>Account</title>" in response.data

        # update account details
        form = UpdateAccountForm()
        form.username.data = "testuserUpdated"
        form.email.data = "updated@gmail.com"
        form.picture.data = "default.jpg"
        response = client.post("/account", data=form.data, follow_redirects=True)
        assert b"Your account has been updated!" in response.data

        # test if user is updated
        user = User.query.filter_by(username="testuserUpdated").first()
        assert user is not None
        assert user.email == "updated@gmail.com"
        assert user.username == "testuserUpdated"


def test_admin_page(client, app):
    with app.app_context():
        # test get request without login
        response = client.get("/admin", follow_redirects=True)
        assert b"Please log in to access this page" in response.data

        # test get request with login
        add_user_and_login(client)
        response = client.get("/admin")
        assert response.status_code == 403

        # logout
        response = client.get("/logout", follow_redirects=True)
        assert b"You have been logged out" in response.data

        # login with admin
        add_admin_user_and_login(client)
        response = client.get("/admin")
        assert response.status_code == 200
        assert b"Admin Home" in response.data


def test_errors(client, app):
    with app.app_context():
        # test get request to non existing page
        response = client.get("/abcd")
        assert response.status_code == 404
        assert b"Page Not Found (404)" in response.data

        # test get request to admin page without admin login
        add_user_and_login(client)
        response = client.get("/admin")
        assert response.status_code == 403
        assert b"You don't have permission to do that (403)" in response.data


def test_add_update_remove_question(client, app):
    with app.app_context():
        # test get request without login
        response = client.get("/admin/add_question", follow_redirects=True)
        assert b"Please log in to access this page" in response.data

        # test get request with login
        add_user_and_login(client)
        response = client.get("/admin/add_question", follow_redirects=True)
        assert response.status_code == 403
        assert b"You don't have permission to do that (403)" in response.data

        # logout
        response = client.get("/logout", follow_redirects=True)

        # login with admin
        add_admin_user_and_login(client)
        response = client.get("/admin/add_question")
        assert response.status_code == 200
        assert b"Question Form" in response.data

        # check the no of questions in the database is zero
        assert Questions.query.count() == 0

        # add a question
        form = AddQuestionForm()
        form.question.data = "What is the capital of India?"
        form.answer.data = "New Delhi"
        form.option_a.data = "New Delhi"
        form.option_b.data = "Mumbai"
        form.option_c.data = "Kolkata"
        form.option_d.data = "Chennai"
        form.section.data = "General Knowledge"
        form.topic.data = "India"
        response = client.post("/admin/add_question", data=form.data)
        assert response.status_code == 302  # redirect data
        assert Questions.query.count() == 1

        # get the question
        question = Questions.query.filter_by(
            Question="What is the capital of India?"
        ).first()
        assert question is not None
        assert question.Answer == "New Delhi"
        assert question.OptionA == "New Delhi"
        assert question.OptionB == "Mumbai"
        assert question.OptionC == "Kolkata"
        assert question.OptionD == "Chennai"
        assert question.Section == "General Knowledge"
        assert question.Topic == "India"

        # update the question
        QID = question.QID
        response = client.get(f"/admin/update_question/{QID}", follow_redirects=True)
        assert response.status_code == 200

        form = UpdateQuestionForm()
        form.question.data = "What is the capital of Chaina?"
        form.answer.data = "Beijing"
        form.option_a.data = "Beijing"
        form.option_b.data = "Shanghai"
        form.option_c.data = "Guangzhou"
        form.option_d.data = "Shenzhen"
        form.section.data = "General Knowledge"
        form.topic.data = "Chaina"
        response = client.post(f"/admin/update_question/{QID}", data=form.data)
        assert response.status_code == 302  # redirect data
        assert Questions.query.count() == 1

        # get the updated question
        question = Questions.query.filter_by(
            Question="What is the capital of Chaina?"
        ).first()
        assert question is not None
        assert question.Answer == "Beijing"
        assert question.OptionA == "Beijing"
        assert question.OptionB == "Shanghai"
        assert question.OptionC == "Guangzhou"
        assert question.OptionD == "Shenzhen"
        assert QID == question.QID

        # remove the question
        response = client.get(f"/admin/remove/{QID}", follow_redirects=True)
        assert response.status_code == 200
        question = Questions.query.filter_by(
            Question="What is the capital of Chaina?"
        ).first()
        assert question is None
        assert Questions.query.count() == 0


def test_logout_page(client, app):
    with app.app_context():
        # test get request with login
        add_user_and_login(client)
        response = client.get("/topics", follow_redirects=True)
        assert b"Topics" in response.data

        # logout
        response = client.get("/logout", follow_redirects=True)
        assert b"You have been logged out" in response.data

        # test get request without login
        response = client.get("/topics", follow_redirects=True)
        assert b"Topics" not in response.data
        assert b"Please log in to access this page" in response.data
        assert b"Login" in response.data


def test_passord_reset(client, app):
    pass


def test_questions_page(client, app):
    pass


def test_topics_page(client, app):
    pass


def test_answer_storage(client, app):
    pass


def test_answer_retieval(client, app):
    pass


def test_multi_user(client, app):
    pass

