import unittest
from unittest.mock import Mock, patch
import contextlib
import io
import tempfile
from pathlib import Path
from color_selection import paint_selected_colors, ColorProgressControl
from color_profile import ColorProfile
from painting_session import Console, PauseRequested


class SelectionTests(unittest.TestCase):
    def execute(self, replies, results=None, save_progress=True):
        plan={'modules':{'color':{'strokes':[{'color':'blue'},{'color':'red'}],
                                  'motion_candidate':ColorProfile(press_mm=.22).to_dict()}}}
        plan['save_progress']=save_progress
        answers=iter(replies)
        runner=Mock(side_effect=results or [0,0])
        station=Mock(return_value=object())
        with patch('run_color_experiment.module_lines', return_value=([{'id':'test'}],[])), contextlib.redirect_stdout(io.StringIO()):
            code=paint_selected_colors(plan,ask=lambda _: next(answers),runner=runner,station_reader=station)
        return code,runner,station

    def test_operator_chooses_order(self):
        code,runner,station=self.execute(['红','blue'])
        self.assertEqual(code,0)
        self.assertEqual([c.kwargs['color'] for c in runner.call_args_list],['red','blue'])

    def test_color_only_entry_honors_no_progress_plan(self):
        code,runner,_=self.execute(['red','blue'],save_progress=False)
        self.assertEqual(code,0)
        self.assertTrue(all(c.kwargs['save_progress'] is False for c in runner.call_args_list))

    def test_live_progress_visible_and_motion_chatter_hidden(self):
        stream=io.StringIO()
        answers=iter(['blue'])
        console=Console(stream,ask=lambda _:next(answers),poll=lambda:False)
        plan={'save_progress':False,'modules':{'color':{'strokes':[{'color':'blue'}],
              'motion_candidate':ColorProfile(press_mm=.22).to_dict()}}}
        def runner(**kw):
            self.assertFalse(kw['save_progress'])
            print('MOVING hidden command')
            for i in (0,1,2):kw['control'].progress(i)
            return 0
        with patch('run_color_experiment.module_lines',return_value=([{},{}],[])), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(paint_selected_colors(plan,ask=lambda _:next(answers),
                runner=runner,station_reader=lambda _:object(),console=console),0)
        self.assertIn('1/2 50.0%',stream.getvalue())
        self.assertIn('2/2 100.0%',stream.getvalue())
        self.assertIn('\r蓝色 [',stream.getvalue())
        self.assertNotIn('MOVING',stream.getvalue())

    def test_progress_pause_parks_before_prompt_and_verifies_continue(self):
        events=[]
        console=Console(io.StringIO(),ask=lambda _:events.append('ask') or 'C',poll=lambda:True)
        control=ColorProgressControl(console,'蓝色',3)
        control.progress(1)
        self.assertTrue(control.boundary(lambda:events.append('park'),lambda:events.append('verify')))
        self.assertEqual(events,['park','ask','verify'])
        self.assertEqual(control.completed,1)

    def test_progress_quit_and_failed_park(self):
        ask=Mock(return_value='Q')
        console=Console(io.StringIO(),ask=ask,poll=lambda:True)
        control=ColorProgressControl(console,'蓝色',3)
        with self.assertRaises(RuntimeError):
            control.boundary(Mock(side_effect=RuntimeError('not parked')),Mock())
        ask.assert_not_called()
        verify=Mock()
        with self.assertRaises(PauseRequested):control.boundary(Mock(),verify)
        verify.assert_not_called()

    def test_no_input_never_moves(self):
        _,runner,station=self.execute([''])
        runner.assert_not_called()
        station.assert_not_called()

    def test_waits_after_one_color(self):
        _,runner,station=self.execute(['red',''])
        self.assertEqual(runner.call_count,1)
        station.assert_called_once_with('red')

    def test_completed_color_not_repeated(self):
        _,runner,_=self.execute(['red','red','blue'])
        self.assertEqual(runner.call_count,2)

    def test_unknown_color_does_not_move(self):
        _,runner,_=self.execute(['purple',''])
        runner.assert_not_called()

    def test_failure_stops_next_color(self):
        code,runner,_=self.execute(['red'],[4])
        self.assertEqual(code,4)
        self.assertEqual(runner.call_count,1)

    def test_black_command_without_execute_hands_off_to_color(self):
        import run_mountain_balanced as black
        # Import before mocking the black loader; otherwise this module keeps
        # a reference to the test mock after the patch exits.
        import run_color_from_job
        with tempfile.TemporaryDirectory() as folder:
            ink=Path(folder)/'sample_黑线任务.json'
            color=Path(folder)/'sample_计划.json'
            color.write_text('{}')
            with patch.object(black,'load_reviewed',return_value=([],None,None,{})), \
                 patch.object(black,'execute_reviewed_job',return_value=0) as draw, \
                 patch('builtins.input',return_value='RUN'), \
                 patch('run_color_from_job.load_plan',return_value={'reviewed':True}) as load, \
                 patch('color_selection.paint_selected_colors',return_value=0) as paint, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(black.main(['--job',str(ink)]),0)
                draw.assert_called_once_with(str(ink))
                load.assert_called_once_with(color,ink)
                paint.assert_called_once_with({'reviewed':True})

    def test_black_cancel_does_not_execute(self):
        import run_mountain_balanced as black
        with patch.object(black,'load_reviewed',return_value=([],None,None,{})), \
             patch.object(black,'execute_reviewed_job') as draw, \
             patch('builtins.input',return_value=''), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(black.main(['--job','unused']),0)
            draw.assert_not_called()


if __name__=='__main__': unittest.main()
