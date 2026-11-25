# -*- coding: utf-8 -*-
import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from drawing_budget_core_20261105 import select_budgets

def row(i,phase=2,module=0,long=False):
    return dict(points=[[2.,float(i*3+2)],[35. if long else 7.,float(i*3+2)]],
        phase=phase,module=module,source_support=.8,source_curve_ids=[i])

class Budgets(unittest.TestCase):
    def test_nested_selection_preserves_structure_long_texture_and_all_modules(self):
        rows=[row(0,0),row(1,1,long=True)]+[row(i,module=i%3) for i in range(2,30)]
        out=select_budgets(rows,(8,15,22))
        small=set(out[8]['selected_ids']);medium=set(out[15]['selected_ids']);large=set(out[22]['selected_ids'])
        self.assertTrue(small<=medium<=large);self.assertTrue({0,1}<=small)
        self.assertEqual({rows[i]['module'] for i in small},{0,1,2})
        self.assertEqual([len(out[b]['selected_ids']) for b in (8,15,22)],[8,15,22])

    def test_budget_yields_to_protected_paths_and_reports_overflow(self):
        rows=[row(i,0) for i in range(10)];out=select_budgets(rows,(3,5,8))
        self.assertTrue(all(r['actual_count']==10 and r['budget_exceeded_for_structure'] for r in out.values()))

    def test_no_padding_no_reconstruction_and_no_input_mutation(self):
        rows=[row(i) for i in range(5)];original=copy.deepcopy(rows);out=select_budgets(rows,(3,8,10))
        self.assertEqual(rows,original);self.assertEqual(out[10]['actual_count'],5)
        self.assertTrue(out[10]['budget_not_filled_due_to_input_size'])
        for r in out.values():
            self.assertEqual(sorted(r['selected_ids']+r['omitted_ids']),list(range(5)))
            self.assertFalse(r['selection_is_expert_verified'])

    def test_source_supported_short_details_are_not_all_removed_by_length(self):
        rows=[row(i,0) for i in range(10)]
        rows += [dict(points=[[0.,float(i)],[9.,float(i)]],phase=1,module=0,source_support=.8) for i in range(10,70)]
        rows += [dict(points=[[20.,float(i)],[23.,float(i)]],phase=2,module=0,source_support=.9) for i in range(70,80)]
        out=select_budgets(rows,(30,50,70))
        self.assertGreaterEqual(out[50]['retained_short_detail_candidates'],5)

if __name__=='__main__':unittest.main()
