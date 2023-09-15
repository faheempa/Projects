import pytest
from project import create_app, db

@pytest.fixture
def app():
    app = create_app(db_url="sqlite:///")
    with app.app_context():
        db.create_all()
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    yield app


@pytest.fixture
def client(app):
    return app.test_client()
