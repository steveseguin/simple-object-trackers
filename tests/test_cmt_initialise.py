"""Focused CMT initialization regressions using real ORB descriptors.

Run with a source-compatible NumPy/OpenCV/SciPy installation:
    python -m unittest discover -s tests -v

Newer OpenCV bindings return keypoint tuples where this legacy tracker expects
lists. The adapter below normalizes that container only; detection, descriptor
computation, and CMT initialization execute their actual implementations.
"""
import os
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cmt-tracker'))
import CMT


class ListKeypointORB:
    def __init__(self, orb):
        self.orb = orb

    def detect(self, image):
        return list(self.orb.detect(image))

    def compute(self, image, keypoints):
        return self.orb.compute(image, keypoints)


class InitialisationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.noise = np.random.RandomState(7).randint(
            0, 256, (240, 320)).astype(np.uint8)

    def setUp(self):
        create_orb = cv2.ORB_create
        self.orb_patch = mock.patch.object(
            CMT.cv2, 'ORB_create',
            side_effect=lambda: ListKeypointORB(create_orb()))
        self.orb_patch.start()
        self.addCleanup(self.orb_patch.stop)

    def assert_database_consistent(self, tracker, has_background):
        count = tracker.num_initial_keypoints
        self.assertGreater(count, 1)
        self.assertEqual(tracker.features_database.dtype,
                         tracker.selected_features.dtype)
        self.assertEqual(tracker.features_database.shape[0],
                         tracker.database_classes.shape[0])
        np.testing.assert_array_equal(
            tracker.features_database[-count:], tracker.selected_features)
        np.testing.assert_array_equal(
            tracker.database_classes[-count:], np.arange(1, count + 1))
        self.assertEqual(tracker.active_keypoints.shape, (count, 3))
        np.testing.assert_array_equal(
            tracker.active_keypoints[:, 2], np.arange(1, count + 1))
        background_count = tracker.features_database.shape[0] - count
        if has_background:
            self.assertGreater(background_count, 0)
            np.testing.assert_array_equal(
                tracker.database_classes[:background_count],
                np.zeros(background_count))
        else:
            self.assertEqual(background_count, 0)

    def test_full_frame_selection_has_no_background(self):
        tracker = CMT.CMT()
        tracker.initialise(self.noise, (0, 0), (320, 240))
        self.assert_database_consistent(tracker, False)

    def test_textured_object_on_plain_background(self):
        image = np.zeros_like(self.noise)
        image[80:160, 120:200] = self.noise[80:160, 120:200]
        tracker = CMT.CMT()
        tracker.initialise(image, (60, 40), (260, 200))
        self.assert_database_consistent(tracker, False)

    def test_mixed_foreground_and_background_are_preserved(self):
        tracker = CMT.CMT()
        tracker.initialise(self.noise, (0, 0), (160, 240))
        self.assert_database_consistent(tracker, True)

    def test_reinitialisation_discards_previous_background(self):
        tracker = CMT.CMT()
        tracker.initialise(self.noise, (0, 0), (160, 240))
        self.assert_database_consistent(tracker, True)
        tracker.initialise(self.noise, (0, 0), (320, 240))
        self.assert_database_consistent(tracker, False)

    def test_empty_selection_still_reports_no_keypoints(self):
        tracker = CMT.CMT()
        with self.assertRaisesRegex(Exception, 'No keypoints found in selection'):
            tracker.initialise(self.noise, (0, 0), (10, 10))


if __name__ == '__main__':
    unittest.main()
