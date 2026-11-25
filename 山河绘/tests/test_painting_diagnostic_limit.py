"""The diagnostic limit must park at a verified stroke boundary."""
import unittest
from unittest.mock import MagicMock

from painting_session import StageControl, PauseRequested
from scripts.run_painting_diagnostic import DiagnosticConsole


class DiagnosticLimitTests(unittest.TestCase):
    def test_limit_parks_before_exit_and_never_requests_resume(self):
        console = DiagnosticConsole(122)
        console.completed = 121
        self.assertFalse(console.poll())
        console.completed = 122
        session = MagicMock()
        session.data = {'manifest': {'stages': {'black': [{}]*146}}}
        session.save_progress = False
        control = StageControl(session, 'black', 'black', console)
        park, verify = MagicMock(), MagicMock()
        with self.assertRaises(PauseRequested):
            control.boundary(park, verify)
        park.assert_called_once()
        session.parked.assert_called_once()
        verify.assert_not_called()

    def test_failed_park_cannot_claim_pause_or_continue(self):
        console = DiagnosticConsole(122)
        console.completed = 122
        session = MagicMock()
        session.data = {'manifest': {'stages': {'black': [{}]*146}}}
        control = StageControl(session, 'black', 'black', console)
        park = MagicMock(side_effect=RuntimeError('alarm'))
        with self.assertRaisesRegex(RuntimeError, 'alarm'):
            control.boundary(park, MagicMock())
        session.parked.assert_not_called()
