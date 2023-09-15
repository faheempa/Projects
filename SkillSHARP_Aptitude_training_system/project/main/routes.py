from flask import (
    Response,
    render_template,
    Blueprint,
    flash,
    redirect,
    url_for,
    request,
    abort,
)
from flask_login import current_user, login_required
from project.models import Questions, QuestionsAnswered, Mock
from project import db


main = Blueprint("main", __name__)


@main.route("/")
@main.route("/home")
def home():
    return render_template("main/home.html", title="Home")


@main.route("/about")
def about():
    return render_template("main/about.html", title="About")


@main.route("/topics")
@login_required
def topics():
    sections = [
        x[0] for x in Questions.query.with_entities(Questions.Section).distinct().all()
    ]
    topics = {}
    for section in sections:
        topics[section] = [
            x[0]
            for x in Questions.query.with_entities(Questions.Topic)
            .filter_by(Section=section)
            .distinct()
            .all()
        ]
    return render_template("main/topics.html", title="Topics", topics=topics)


@main.route("/questions/<string:section>/<string:topic>")
@login_required
def questions(section, topic):
    s, t = section.replace("_", " "), topic.replace("_", " ")
    questions = Questions.query.filter_by(Section=s, Topic=t).all()
    answers = QuestionsAnswered.query.filter_by(UID=current_user.id).all()
    answers = {x.QID: (x.Option, x.Status) for x in answers}
    return render_template(
        "main/questions.html",
        title=t,
        questions=questions,
        count=1,
        len=len(questions),
        answers=answers,
    )


@main.route("/save/<string:qid>/<string:status>/<string:selection>")
@login_required
def save(qid, status, selection):
    uid = current_user.id

    data = QuestionsAnswered.query.filter_by(QID=qid, UID=uid).first()
    if data != None:
        data.Status = status
        data.Option = selection
    else:
        data = QuestionsAnswered(
            UID=current_user.id, QID=qid, Status=status, Option=selection
        )

    db.session.add(data)
    db.session.commit()
    return Response(status=204)


@main.route("/mocktest")
@login_required
def mocktest():
    return render_template("main/mocktest.html", title="Mock Test")


@main.route("/mocktest/<string:level>")
@login_required
def mocktest_level(level):
    questions = (
        Questions.query.filter_by(Level=level)  
        .order_by(db.func.random())
        .limit(10)
        .all()
    )
    return render_template(
        "main/mocktest_questions.html",
        title=f"Mock Test - {level}",
        questions=questions,
        count=1,
        len=len(questions),
    )


@main.route("/mock/<string:level>/<string:score>/<string:time>")
@login_required
def save_mocktest(level, score, time):
    data = Mock(UID=current_user.id, Level=level, Score=score, Time=time)
    db.session.add(data)
    db.session.commit()
    return Response(status=204)

