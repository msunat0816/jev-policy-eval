"""Semantic boundary checks; no model calls and no video dependencies."""
import unittest

from measurement_adapter import adapt_requirement, requirement_states, requirement_verdict


LAYOUT_TASK = ("At episode end, the bounding box of red block must be entirely to the left of "
               "the bounding box of blue target in the image plane.")
MOTION_TASK = ("Over the entire episode, the maximum image-plane relative-center excursion of red block "
               "relative to blue target, measured from the first frame, must be at most 1% of the image diagonal.")


def requirement(task=LAYOUT_TASK):
    return {"id": "r0", "text": task, "subject": "red_block", "reference": "blue_target"}


class MeasurementAdapterTests(unittest.TestCase):
    def test_full_explicit_instruction_can_be_linked_but_physical_or_extra_constraints_cannot(self):
        req = requirement()
        good = adapt_requirement(LAYOUT_TASK, req, "layout", [2, 4], 4)
        self.assertEqual(good["coverage"], "direct")
        for task in ("Put the red block on the blue target.",
                     "Put the red block to the left of the blue target.",
                     "Do not " + LAYOUT_TASK,
                     LAYOUT_TASK + " Release the gripper.",
                     "If the gripper opens, " + LAYOUT_TASK):
            with self.subTest(task=task):
                result = adapt_requirement(task, req, "layout", [2, 4], 4)
                self.assertEqual(result["coverage"], "not_established")
                self.assertEqual(requirement_verdict(result, {}, 3), "unknown")

    def test_no_predicate_alias_attribute_direction_or_requirement_rewrite_bypass(self):
        for changes in ({"subject": "block"}, {"reference": "other_target"},
                        {"text": LAYOUT_TASK.replace("left", "right")},
                        {"text": "The block is left of the target.", "predicate": "image_left"}):
            with self.subTest(changes=changes):
                req = dict(requirement(), **changes)
                self.assertNotEqual(adapt_requirement(LAYOUT_TASK, req, "layout", [2, 4], 4)["coverage"], "direct")

    def test_terminal_layout_scope_and_pixel_deadband(self):
        req = requirement()
        early = adapt_requirement(LAYOUT_TASK, req, "layout", [0, 2], 4)
        self.assertEqual(early["coverage"], "partial")
        self.assertEqual(requirement_verdict(early, {"end_box_gaps_px": {"image_left": 20}}, 3), "unknown")
        final = adapt_requirement(LAYOUT_TASK, req, "layout", [2, 4], 4)
        for gap, expected in ((3, "supported"), (-3, "contradicted"), (2, "unknown"), (-2, "unknown")):
            with self.subTest(gap=gap):
                self.assertEqual(requirement_verdict(final, {"end_box_gaps_px": {"image_left": gap}}, 3), expected)

    def test_motion_uses_explicit_numeric_limit_and_entire_episode_scope(self):
        req = requirement(MOTION_TASK)
        local = adapt_requirement(MOTION_TASK, req, "motion", [2, 4], 4)
        full = adapt_requirement(MOTION_TASK, req, "motion", [0, 4], 4)
        value = {"fraction_of_image_diagonal": 0.005}
        self.assertEqual(requirement_verdict(local, value, 3), "unknown")
        self.assertEqual(requirement_verdict(full, value, 5), "supported")
        self.assertEqual(requirement_verdict(full, {"fraction_of_image_diagonal": .02}, 5), "contradicted")
        self.assertEqual(requirement_verdict(full, value, 1), "unknown")
        reverse_task = MOTION_TASK.replace("at most", "at least")
        reverse = adapt_requirement(reverse_task, requirement(reverse_task), "motion", [0, 4], 4)
        self.assertEqual(requirement_verdict(reverse, value, 5), "contradicted")
        physical = "Keep the red block stable on the blue target."
        self.assertNotEqual(adapt_requirement(physical, requirement(physical), "motion", [0, 4], 4)["coverage"], "direct")

    def test_wrong_expert_is_not_a_direct_requirement_check(self):
        result = adapt_requirement(LAYOUT_TASK, requirement(), "motion", [0, 4], 4)
        self.assertEqual(result["reason"], "expert_does_not_measure_required_quantity")
        self.assertEqual(requirement_verdict(result, {}, 5), "unknown")

    def test_latest_direct_contradiction_unknown_and_conflict_replace_earlier_support(self):
        def record(name, last, status):
            return {"id": name, "requirement_id": "r0", "frames": [0, last],
                    "semantic_adapter": {"coverage": "direct"}, "requirement_status": status}
        early = record("early", 2, "supported")
        for final_status in ("contradicted", "unknown"):
            records = [early, record("late", 4, final_status)]
            self.assertEqual(requirement_states([requirement()], records)[0]["status"], final_status)
        conflict = [record("a", 4, "supported"), record("b", 4, "contradicted")]
        self.assertEqual(requirement_states([requirement()], conflict)[0]["reason"], "conflicting_direct_evidence")
        # An inapplicable expert does not invalidate an independent, complete direct check.
        same_time = [record("a", 4, "supported"), record("b", 4, "unknown")]
        self.assertEqual(requirement_states([requirement()], same_time)[0]["status"], "supported")


if __name__ == "__main__":
    unittest.main()
