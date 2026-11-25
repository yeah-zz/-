import unittest
from gongbi_training.cunfa_evidence_levels import supervision_for,IGNORE

class EvidenceChecks(unittest.TestCase):
    def test_example_style_never_creates_pixel_roles(self):
        r=supervision_for(dict(kind='named_example',caption_checked=True,source_group='one_artist',label_scope='whole_teaching_example',initial_taxonomy='hemp'))
        self.assertEqual(r['image_style_target'],0);self.assertFalse(r['pixel_role_enabled'])
    def test_video_role_never_creates_style(self):
        r=supervision_for(dict(kind='video_supported_band',assistant_reviewed=True,stage_pair_checked=True,partial_mask_sha256='hash',named_style=None))
        self.assertTrue(r['pixel_role_enabled']);self.assertEqual(r['image_style_target'],IGNORE)
    def test_tree_cannot_be_cunfa(self):
        with self.assertRaises(ValueError):supervision_for(dict(kind='named_example',object_type='tree',cunfa_target='hemp'))
    def test_title_only_is_insufficient(self):
        with self.assertRaises(ValueError):supervision_for(dict(kind='named_example',caption_checked=False))
    def test_out_of_scope_is_known_category_not_missing_label(self):
        r=supervision_for(dict(kind='named_example',caption_checked=True,source_group='one_artist',label_scope='whole_teaching_example',initial_taxonomy='known_out_of_scope'))
        self.assertEqual(r['image_style_target'],2)

if __name__=='__main__':unittest.main()
