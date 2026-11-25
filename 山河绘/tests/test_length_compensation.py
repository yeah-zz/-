import unittest
from dataclasses import replace
from pigment_station import PigmentStation

class DistanceCompensationTests(unittest.TestCase):
 def test_boundaries_caps_and_count_independence(self):
  s=PigmentStation(0,0,0,0,level_compensation_enabled=True,level_every_length_mm=500,level_step_mm=.1,level_max_mm=.5)
  for length,expected in [(0,0),(499.99,0),(500,.1),(999.99,.1),(1000,.2),(2500,.5),(9000,.5)]:
   self.assertAlmostEqual(s.level_offset(999,length),expected)
   self.assertAlmostEqual(s.level_offset(0,length),expected)
  self.assertEqual(replace(s,level_compensation_enabled=False).level_offset(999,9000),0)
  for value in [-1,float('nan'),float('inf')]:
   with self.assertRaises(ValueError):s.level_offset(0,value)
 def test_new_session_resets(self):
  s=PigmentStation(0,0,0,0,level_compensation_enabled=True,level_every_length_mm=500,level_step_mm=.1,level_max_mm=.5)
  self.assertEqual(s.level_offset(0,0),0)
