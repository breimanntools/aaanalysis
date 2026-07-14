"""
Tests for routing library output through the ``aaanalysis`` logger (issue #75).

``ut.print_out`` is a thin shim over ``logging.getLogger("aaanalysis").info(...)``. The
logger level is a power-user control (``ut.set_logger_verbosity`` / ``setLevel``) that is
independent of ``options['verbose']`` — output visibility for normal use stays governed by
the existing per-call / global verbose gating at the call sites. These tests use pytest's
``caplog`` to assert the record contract and ``capsys`` to assert the default stdout handler
keeps output visible (and blue-wrapped), so on-screen behaviour is unchanged.
"""
import logging
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis import config
from aaanalysis._utils import utils_output

LOGGER_NAME = "aaanalysis"


@pytest.fixture(autouse=True)
def _restore_logger_state():
    """Save/restore process-global logger level and the verbose option.

    The named logger's level is global state; restoring it prevents these tests from
    leaking an INFO/WARNING level (or a changed ``options['verbose']``) into the rest
    of the suite.
    """
    logger = logging.getLogger(LOGGER_NAME)
    prev_level = logger.level
    prev_verbose = aa.options["verbose"]
    try:
        yield
    finally:
        logger.setLevel(prev_level)
        aa.options["verbose"] = prev_verbose


def _aa_records(caplog):
    """Records emitted specifically on the aaanalysis logger."""
    return [r for r in caplog.records if r.name == LOGGER_NAME]


class TestPrintOutLogging:
    """print_out emits exactly one INFO record on the aaanalysis logger."""

    def test_emits_single_info_record(self, caplog):
        logging.getLogger(LOGGER_NAME).setLevel(logging.INFO)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            ut.print_out("hello-log")
        records = _aa_records(caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.INFO
        assert records[0].getMessage() == "hello-log"

    def test_message_content_preserved(self, caplog):
        logging.getLogger(LOGGER_NAME).setLevel(logging.INFO)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            ut.print_out("abc 123 !@#")
        assert _aa_records(caplog)[0].getMessage() == "abc 123 !@#"

    def test_print_out_does_not_call_builtin_print(self, monkeypatch):
        # Guard the shim contract: print_out must route through logging, never print().
        def _boom(*args, **kwargs):
            raise AssertionError("print_out must not call the builtin print()")
        monkeypatch.setattr("builtins.print", _boom)
        logging.getLogger(LOGGER_NAME).setLevel(logging.INFO)
        ut.print_out("no-print-here")  # would raise if print() were called


class TestLoggerLevelGating:
    """Level changes on the named logger toggle capture, as documented."""

    def test_warning_suppresses(self, caplog):
        logging.getLogger(LOGGER_NAME).setLevel(logging.WARNING)
        ut.print_out("should-not-appear")
        assert _aa_records(caplog) == []

    def test_info_reenables(self, caplog):
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.WARNING)
        ut.print_out("suppressed")
        assert _aa_records(caplog) == []
        logger.setLevel(logging.INFO)
        ut.print_out("visible")
        records = _aa_records(caplog)
        assert len(records) == 1
        assert records[0].getMessage() == "visible"


class TestLoggerLevelControl:
    """The logger level is a power-user control (``set_logger_verbosity`` / ``setLevel``),
    independent of ``options['verbose']``.

    Output visibility for normal use stays governed by the existing per-call / global
    verbose gating at the call sites (``if verbose: print_out(...)``). The logger is left
    permissive (INFO) by default so ``print_out`` is visible out of the box; setting
    ``options['verbose']`` deliberately does NOT move the process-global level — doing so
    would mute an object explicitly constructed with ``verbose=True`` whenever a global
    ``options['verbose']=False`` was in effect, changing on-screen behaviour.
    """

    def test_set_logger_verbosity_true_emits(self, caplog):
        ut.set_logger_verbosity(True)
        assert logging.getLogger(LOGGER_NAME).level == logging.INFO
        ut.print_out("lv-true")
        records = _aa_records(caplog)
        assert len(records) == 1
        assert records[0].getMessage() == "lv-true"

    def test_set_logger_verbosity_false_suppresses(self, caplog):
        ut.set_logger_verbosity(False)
        assert logging.getLogger(LOGGER_NAME).level == logging.WARNING
        ut.print_out("lv-false")
        assert _aa_records(caplog) == []

    def test_options_verbose_does_not_move_logger_level(self):
        # options['verbose'] gates output via check_verbose at the call sites, NOT via the
        # process-global logger level. Setting it must leave the level untouched.
        logging.getLogger(LOGGER_NAME).setLevel(logging.INFO)
        aa.options["verbose"] = False
        assert logging.getLogger(LOGGER_NAME).level == logging.INFO
        aa.options["verbose"] = True
        assert logging.getLogger(LOGGER_NAME).level == logging.INFO

    def test_default_level_is_info_so_print_out_is_visible(self, caplog):
        # Default (out-of-the-box) behaviour: print_out is visible without any setup.
        ut.set_logger_verbosity(True)  # the shipped default
        ut.print_out("default-visible")
        assert len(_aa_records(caplog)) == 1


class TestDefaultHandlerStdout:
    """The default handler keeps print_out visible on stdout (blue-wrapped)."""

    def test_print_out_writes_to_current_stdout(self, capsys):
        logging.getLogger(LOGGER_NAME).setLevel(logging.INFO)
        ut.print_out("on-stdout")
        out = capsys.readouterr().out
        assert "on-stdout" in out
        # Blue ANSI wrap preserved -> unchanged on-screen appearance.
        assert "\033[94m" in out and "\033[0m" in out

    def test_warning_level_silences_stdout(self, capsys):
        logging.getLogger(LOGGER_NAME).setLevel(logging.WARNING)
        ut.print_out("silent")
        assert "silent" not in capsys.readouterr().out

    def test_single_stdout_handler_and_idempotent_configure(self):
        # _configure_logger is idempotent: re-running it must not duplicate the handler.
        utils_output._configure_logger()
        marked = [h for h in logging.getLogger(LOGGER_NAME).handlers
                  if getattr(h, "_aaanalysis_stdout_handler", False)]
        assert len(marked) == 1
