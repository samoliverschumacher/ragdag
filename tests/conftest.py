from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def mock_write_log_to_db():
    with mock.patch("app.logconfig.write_data") as mock_write_log_to_db:
        yield mock_write_log_to_db
