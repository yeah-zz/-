import unittest
from run_color_experiment import long_stroke_reload_policy,color_reload_limits,needs_color_reload

POLICY=dict(min_length_mm=30.,very_long_length_mm=50.,strokes_per_load=3,
            very_long_strokes_per_load=2,max_length_mm=160.)

def batches(lengths,policy=POLICY,default_limit=8):
    groups=[];batch=[];loaded=None;used=0.;largest=0.
    for i,length in enumerate(lengths):
        limit,cap=color_reload_limits(policy,length,largest,default_limit)
        if needs_color_reload(loaded,len(batch),used,length,i,limit,length_limit_mm=cap):
            if batch:groups.append(batch)
            batch=[];used=0.;largest=0.;loaded=i
        batch.append(length);used+=length;largest=max(largest,length)
    if batch:groups.append(batch)
    return groups

class LongStrokeReloadTests(unittest.TestCase):
    def test_eight_six_four_refill(self):
        policy=dict(format='three-length-classes-v1',short_below_mm=10.,long_above_mm=20.,
                    short_strokes=8,medium_strokes=6,long_strokes=4,max_length_mm=160.)
        self.assertEqual(long_stroke_reload_policy({'long_stroke_reload':policy}),policy)
        for length,count in [(9.999,8),(10.,6),(20.,6),(20.001,4)]:
            self.assertEqual([len(x) for x in batches([length]*(count+1),policy,8)],[count,1])
        self.assertEqual(batches([5.,10.,5.,5.,5.,5.,5.],policy,8),[[5.,10.,5.,5.,5.,5.],[5.]])
        self.assertEqual(batches([21.,5.,5.,5.,5.],policy,8),[[21.,5.,5.,5.],[5.]])
        self.assertTrue(needs_color_reload(0,1,90,90,1,4,length_limit_mm=160))

    def test_three_classes_boundaries_and_mixed_batches(self):
        policy=dict(format='three-length-classes-v1',short_below_mm=10.,
                    long_above_mm=20.,short_strokes=6,medium_strokes=4,
                    long_strokes=2,max_length_mm=160.)
        self.assertEqual(long_stroke_reload_policy({'long_stroke_reload':policy}),policy)
        for length,count in [(9.999,6),(10.,4),(20.,4),(20.001,2)]:
            self.assertEqual([len(x) for x in batches([length]*(count+1),policy,6)],[count,1])
        self.assertEqual(batches([5.,10.,5.,5.,5.],policy,6),[[5.,10.,5.,5.],[5.]])
        self.assertEqual(batches([5.,21.,5.,5.],policy,6),[[5.,21.],[5.,5.]])
        self.assertEqual(batches([5.,5.,5.,21.],policy,6),[[5.,5.,5.],[21.]])
        for bad in [policy|{'short_below_mm':True},policy|{'medium_strokes':3},
                    policy|{'long_above_mm':float('nan')},policy|{'extra':1}]:
            with self.assertRaises(ValueError):
                long_stroke_reload_policy({'long_stroke_reload':bad})

    def test_latest_six_short_two_long_policy(self):
        policy=POLICY|{'strokes_per_load':2,'short_max_length_mm':180.}
        self.assertEqual([len(x) for x in batches([29.9]*13,policy,6)],[6,6,1])
        for length in [30.,40.,50.,79.9]:
            self.assertEqual([len(x) for x in batches([length]*5,policy,6)],[2,2,1])
        self.assertEqual(batches([60.,60.,5.,5.,5.,5.,5.,5.],policy,6),[[60.,60.],[5.,5.,5.,5.,5.,5.]])
        self.assertEqual(batches([5.,40.,5.,5.,5.,5.,5.,5.],policy,6),[[5.,40.],[5.,5.,5.,5.,5.,5.]])
        self.assertEqual(long_stroke_reload_policy({'long_stroke_reload':policy}),policy)

    def test_medium_contacts_three_per_load_even_above_old_length_cap(self):
        self.assertEqual([len(x) for x in batches([40.]*7)],[3,3,1])
        self.assertEqual([len(x) for x in batches([49.]*6)],[3,3])

    def test_longest_contacts_two_per_load(self):
        for length in [50.,60.,79.9,80.]:
            self.assertEqual([len(x) for x in batches([length]*5)],[2,2,1])

    def test_short_batches_identical_to_original_rule(self):
        for lengths in [[5.]*20,[29.]*10,[5.,15.,29.,1.,8.,10.,14.,2.]*4]:
            self.assertEqual(batches(lengths),batches(lengths,None))
        self.assertEqual([len(x) for x in batches([5.]*17)],[8,8,1])

    def test_mixed_batches_respect_consumption_and_previous_longest(self):
        self.assertEqual(batches([70.,35.,35.,35.,35.]),[[70.,35.],[35.,35.,35.]])
        self.assertEqual(batches([5.,5.,40.,40.]),[[5.,5.,40.],[40.]])
        self.assertEqual(color_reload_limits(POLICY,10,80,8),(8,100.))
        self.assertEqual(color_reload_limits(POLICY,30,0,8),(3,160.))
        self.assertEqual(color_reload_limits(POLICY,50,0,8),(2,160.))

    def test_time_pause_and_cumulative_length_still_trigger(self):
        self.assertTrue(needs_color_reload(None,1,60,60,20,2,length_limit_mm=160))
        self.assertTrue(needs_color_reload(0,1,60,60,61,2,length_limit_mm=160))
        self.assertTrue(needs_color_reload(0,1,90,90,1,2,length_limit_mm=160))
        self.assertFalse(needs_color_reload(0,1,60,60,1,2,length_limit_mm=160))

    def test_policy_is_opt_in_and_validated(self):
        self.assertIsNone(long_stroke_reload_policy({}))
        self.assertEqual(long_stroke_reload_policy({'long_stroke_reload':POLICY}),POLICY)
        for bad in [POLICY|{'min_length_mm':True},POLICY|{'very_long_length_mm':20},POLICY|{'strokes_per_load':8},POLICY|{'max_length_mm':float('nan')},{}]:
            with self.assertRaises(ValueError):long_stroke_reload_policy({'long_stroke_reload':bad})

if __name__=='__main__':unittest.main()
