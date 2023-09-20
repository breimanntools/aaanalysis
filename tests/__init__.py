"""
A summary of different testing strategies is provided as general background:

1. Testing pyramid: Unit test (unit) >> Integration test (integration) >> System/End-to-End test (e2e)
    a) Unit test: Check small bit of code (e.g., function) in isolation
    b) Integration/Regression test: Check a larger bit of code (e.g., several classes)
        Integration with external components/Sequence regression of internal calls
    c) System test: Check whole system in different environments

2. Positive vs negative testing
    a) Positive unit testing: Check if code runs with valid input
    b) Negative testing: Check if code troughs error with invalid input

3. Additional test strategies
    a) Property-Based Testing: Validate assumptions (hypothesis) of code using automatically generated data
        "Complementary to unit testing" (p. 224-230, The Pragmatic Programmer)
    b) Functional test: Check single bit of functionality in a system (similar to regression test?)
        Unit test vs. functional test (Code is doing things right vs. Code is doing right things)

Notes
-----
Recommended testing commands:
    a) General:     pytest -v -p no:warnings --tb=no test_cpp.py {line, short}
    b) Function:    pytest -v -p no:warnings --tb=no test_cpp.py::TestCPP:test_add_stat
    c) Doctest:     pytest -v --doctest-modules -p no:warnings cpp_tools/feature.py
    d) Last failed: pytest --lf

Recommended testing pattern: GIVEN, WHEN, THEN

Recommended testing tools for pytest (given page from Brian, 2017):
    a) Fixtures in conftest file (p. 50)
    b) Parametrized Fixtures (p. 64)
    c) Testing doctest namespace (p. 89)

Following other testing tools are used:
    a) Coverage.py: Determine how much code is tested (via pytest --cov=cpp_tools) (p. 126, Brian, 2017)
    b) tox:         Testing multiple configuration
    c) hypothesis:  Testing tool for property-based testing

References
----------
Brian Okken, Python Testing with pytest, The Pragmatic Programmers (2017)
David Thomas & Andrew Hunt, The Pragmatic Programmer, 20th Anniversary Edition (2019)
    pp. 224-231
David R. Maclver, Zac Hatfield-Dodds, ..., Hypothesis: A new approach to property-based testing (2019)
Harry Percival & Bob Gergory, Architecture Patterns with Python (2020)
"""
