"""Contract checks for model output; fixtures are synthetic, not visual validation."""
import copy
import json
import unittest

from coarse_observation import parse_observation


def observation():
    return {
        "requirements": [{"id": "r0", "text": "Place the beige block on the left peg.",
                          "subject": "beige_block", "reference": "left_peg"}],
        "frames": [
            {"frame": 7, "visible_facts": ["The block is beside the peg in the image."],
             "uncertainties": [], "boxes": {"beige_block": [10, 20, 50, 60],
                                              "left_peg": [80, 20, 120, 60]}},
            {"frame": 27, "visible_facts": ["The block and peg overlap in the image."],
             "uncertainties": ["Support is obscured."], "boxes": {}}
        ],
        "candidate_checks": [{"id": "c0", "requirement_ids": ["r0"],
                              "entity_keys": ["beige_block", "left_peg"],
                              "related_frames": [7, 27],
                              "question": "Is the block supported by the peg?",
                              "rationale": "The sampled positions differ, but support is obscured."}]
    }


class ObservationContractTests(unittest.TestCase):
    def parse(self, value):
        return parse_observation(json.dumps(value), [7, 27])

    def test_preserves_natural_requirement_and_uncertainty_without_predicate(self):
        value = observation()
        self.assertEqual(self.parse(value), value)
        self.assertNotIn("predicate", self.parse(value)["requirements"][0])

    def test_allows_missing_boxes_empty_checks_and_no_separate_reference(self):
        value = observation()
        value["candidate_checks"] = []
        value["requirements"][0]["reference"] = ""
        for frame in value["frames"]:
            frame.update(boxes={}, visible_facts=[], uncertainties=["Object identity unclear."])
        self.assertEqual(self.parse(value), value)

    def test_rejects_missing_duplicate_unprovided_and_reordered_frames(self):
        for ids in ([7], [7, 7], [7, 28], [27, 7], [True, 27]):
            with self.subTest(ids=ids):
                value = observation()
                value["frames"] = [dict(copy.deepcopy(value["frames"][0]), frame=i) for i in ids]
                with self.assertRaises(ValueError):
                    self.parse(value)

    def test_rejects_dangling_requirement_and_frame_references(self):
        for field, refs in (("requirement_ids", ["r9"]), ("related_frames", [8]),
                            ("related_frames", []), ("requirement_ids", [])):
            with self.subTest(field=field, refs=refs):
                value = observation()
                value["candidate_checks"][0][field] = refs
                with self.assertRaises(ValueError):
                    self.parse(value)

    def test_rejects_duplicate_requirement_and_check_ids(self):
        for field in ("requirements", "candidate_checks"):
            with self.subTest(field=field):
                value = observation()
                value[field].append(copy.deepcopy(value[field][0]))
                with self.assertRaises(ValueError):
                    self.parse(value)

    def test_rejects_degenerate_out_of_range_and_non_numeric_boxes(self):
        for box in ([0, 0, 0, 50], [50, 0, 20, 50], [-1, 0, 20, 50],
                    [0, 0, 1001, 50], [False, 0, 20, 50], [0, 0, float("nan"), 50],
                    [0, 0, float("inf"), 50], ["0", 0, 20, 50]):
            with self.subTest(box=box):
                value = observation()
                value["frames"][0]["boxes"]["beige_block"] = box
                with self.assertRaises(ValueError):
                    self.parse(value)

    def test_does_not_require_a_box_for_an_uncertain_actor(self):
        value = observation()
        value["candidate_checks"][0]["entity_keys"].append("occluded_gripper")
        self.assertEqual(self.parse(value), value)

    def test_rejects_old_schema_and_extra_verdict_fields(self):
        for field, content in (("score", 5), ("predicate", "image_above")):
            with self.subTest(field=field):
                value = observation()
                if field == "score":
                    value[field] = content
                else:
                    value["requirements"][0][field] = content
                with self.assertRaises(ValueError):
                    self.parse(value)
        value = observation()
        value["frames"][0]["observation"] = value["frames"][0].pop("visible_facts")
        with self.assertRaises(ValueError):
            self.parse(value)

    def test_rejects_prose_wrapped_json_and_non_string_facts(self):
        with self.assertRaises(ValueError):
            parse_observation("```json\n" + json.dumps(observation()) + "\n```", [7, 27])
        value = observation()
        value["frames"][0]["visible_facts"] = [123]
        with self.assertRaises(ValueError):
            self.parse(value)


if __name__ == "__main__":
    unittest.main()
