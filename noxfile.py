import nox

@nox.session(python=["3.9", "3.10", "3.11", "3.12"])
def tests(session):
    session.install("-r", "requirements.txt")
    session.install("pytest")
    session.run("pytest", "--verbose", "tests/")