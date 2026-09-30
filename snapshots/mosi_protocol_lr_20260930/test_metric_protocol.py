import unittest
import numpy as np
from metric_protocol import classes,audit,binary_metrics

class TestProtocol(unittest.TestCase):
    def test_ties_and_clamp(self):
        x=[-9,-2.5,-1.5,-.5,0,.5,1.5,2.5,9]
        self.assertEqual(classes(x,'legacy_away').tolist(),[-3,-3,-2,-1,0,1,2,3,3])
        self.assertEqual(classes(x,'nearest_even').tolist(),[-3,-2,-2,0,0,0,2,2,3])
    def test_both_directions(self):
        x=audit([.5,.5],[.4,.6])
        self.assertEqual((x['wrong_to_right'],x['right_to_wrong']),(1,1))
        self.assertEqual(x['Acc7_legacy'],x['Acc7_nearest_even'])
    def test_adjacent_float32(self):
        x=np.array([np.nextafter(np.float32(.5),np.float32(0)),np.nextafter(np.float32(.5),np.float32(1))])
        self.assertEqual(classes(x,'nearest_even').tolist(),[0,1])
        # Legacy float32 add(.5) itself rounds the lower adjacent value to1.
        self.assertEqual(classes(x,'legacy_away').tolist(),[1,1])
    def test_f1_and_zero(self):
        m=binary_metrics([-1,1,1,0],[-1,-1,1,-1])
        self.assertEqual(m['nonzero']['n'],3)
        self.assertAlmostEqual(m['all']['macro_F1'],50.)
        self.assertAlmostEqual(m['all']['weighted_F1'],50.)
        self.assertAlmostEqual(m['nonzero']['macro_F1'],200/3)
    def test_reject_invalid(self):
        with self.assertRaises(ValueError):audit([0],[float('nan')])

if __name__=='__main__':unittest.main()
