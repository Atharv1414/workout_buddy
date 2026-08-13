import unittest

from buddy import RepCounter, calculate_angle


class AngleTests(unittest.TestCase):
    def test_right_angle(self):
        self.assertAlmostEqual(calculate_angle((1, 0), (0, 0), (0, 1)), 90)

    def test_straight_angle(self):
        self.assertAlmostEqual(calculate_angle((-1, 0), (0, 0), (1, 0)), 180)


class CounterTests(unittest.TestCase):
    def test_squat_counts_full_cycle_only(self):
        counter = RepCounter("squat")
        for angle in (170, 130, 90, 92, 120, 165):
            counter.update(angle)
        self.assertEqual(counter.reps, 1)

    def test_pushup_does_not_count_partial_rep(self):
        counter = RepCounter("pushup")
        for angle in (165, 130, 120, 160):
            counter.update(angle)
        self.assertEqual(counter.reps, 0)

    def test_curl_counts_contraction(self):
        counter = RepCounter("curl")
        for angle in (160, 100, 50, 48):
            counter.update(angle)
        self.assertEqual(counter.reps, 1)


if __name__ == "__main__":
    unittest.main()
