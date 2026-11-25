import unittest
from gongbi_training.stage_reference_policy import permitted_supervision

class StagePolicyChecks(unittest.TestCase):
    def sample(self):return dict(style_evidence='explicit_mixed_named_object',style_components=['hemp','axe'],artist_attribution_checked=True,ordered_steps_checked=True)
    def test_mixed_not_forced_into_binary(self):
        r=permitted_supervision(self.sample());self.assertEqual(r['knowledge_style_tag'],'mixed');self.assertFalse(r['binary_style_training'])
    def test_article_does_not_create_pixel_or_stroke_truth(self):
        r=permitted_supervision(self.sample());self.assertFalse(r['pixel_role_training']);self.assertFalse(r['single_stroke_training'])
    def test_different_crops_cannot_be_frame_differenced(self):
        self.assertFalse(permitted_supervision(self.sample())['aligned_temporal_difference_allowed'])
    def test_missing_artist_not_reliable_stage_supervision(self):
        s=self.sample();s['artist_attribution_checked']=False;self.assertFalse(permitted_supervision(s)['source_specific_stage_order'])
    def test_source_steps_not_autonomous_planning(self):
        self.assertTrue(permitted_supervision(self.sample())['source_specific_stage_order']);self.assertFalse(permitted_supervision(self.sample())['automatic_planner_verified'])

if __name__=='__main__':unittest.main()
