import json
import logging
import sys

from grader.log import JsonFormatter, configure_logging, get_logger


def test_json_formatter_includes_extras_and_exception():
    formatter = JsonFormatter()
    logger = logging.getLogger("test.json")
    try:
        raise ValueError("boom")
    except ValueError:
        record = logger.makeRecord(
            "test.json",
            logging.ERROR,
            __file__,
            1,
            "failed %s",
            ("x",),
            exc_info=sys.exc_info(),
            extra={"job_id": "abc", "count": 3},
        )
    payload = json.loads(formatter.format(record))
    assert payload["level"] == "ERROR"
    assert payload["msg"] == "failed x"
    assert payload["job_id"] == "abc"
    assert payload["count"] == 3
    assert "ValueError: boom" in payload["exc_info"]
    assert "ts" in payload


def test_configure_logging_is_idempotent():
    configure_logging("DEBUG")
    configure_logging("WARNING")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)
    assert root.level == logging.WARNING
    assert get_logger("x").name == "x"
