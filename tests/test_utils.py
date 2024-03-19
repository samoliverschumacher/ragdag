import time

import pytest

from app.utils import run_until_timeout


def function_fast():
    time.sleep(1)
    return "Finished quickly"

def function_slow():
    time.sleep(10)
    return "Finished slowly"

class CustomTimeoutException(Exception):
    pass

def test_run_with_timeout_no_timeout():
    result = run_until_timeout(function_fast, timeout=2, timeout_error_type=CustomTimeoutException)
    assert result == "Finished quickly"

def test_run_with_timeout_with_timeout():
    with pytest.raises(CustomTimeoutException, match=r".*took too long to respond"):
        run_until_timeout(function_slow, timeout=1, timeout_error_type=CustomTimeoutException)
