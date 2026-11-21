"""Pure matching contracts; synthetic proposals do not validate model judgments."""
import copy
import json
import unittest

from test_window_selection import selection_observation
from vlm_matching import (EXPERT_CATALOG, matching_context, parse_window_matching,
                          validate_matching_inputs)
from window_selection import summarize_selection


def matching_observation():
    obs = selection_observation()
    obs["requirements"][0].update(text="Place the apple in the bowl.", subject="apple", reference="bowl")
    obs["candidate_checks"][0].update(
        entity_keys=["apple", "bowl"], question="Is the apple in the bowl at episode end?",
        rationale="Sampled image overlap is not yet verified containment.")
    obs["frames"][0]["visible_facts"] = ["The apple is beside the bowl."]
    obs["frames"][1]["visible_facts"] = ["The apple and bowl overlap in the image."]
    obs["frames"][2]["uncertainties"] = ["The apple is obscured."]
    return obs


def window_matching(window_id="w0", first=0, last=2, *, motion=True):
    experts = [{"expert_id": "object_spatial", "entity_keys": ["apple", "bowl"],
                "frame_ids": sorted({first, last}), "reason": "Obtain candidate object masks and 2D overlap."}]
    if motion:
        experts.append({"expert_id": "short_window_motion", "entity_keys": ["apple"],
                        "frame_ids": list(range(first, last + 1)),
                        "reason": "Measure local apple displacement after a reliable mask is available."})
    return {"window_id": window_id, "checks": [
        {"check_id": "c0", "relevance": "related", "reason": "The apple and bowl are relevant to containment.",
         "experts": experts, "limitations": ["2D overlap does not establish containment; masks/texture remain prerequisites."]},
        {"check_id": "c1", "relevance": "unrelated", "reason": "No drawer evidence is identified in this window.",
         "experts": [], "limitations": []},
    ]}


class VlmMatchingTests(unittest.TestCase):
    def parse(self, value):
        return parse_window_matching(json.dumps(value), {"id": "w0"},
                                     matching_observation()["candidate_checks"], [0, 1, 2])

    def test_preserves_complete_checklist_and_proposals_without_completion_verdicts(self):
        value = window_matching()
        self.assertEqual(self.parse(value), value)
        context = matching_context("Original instruction", matching_observation(), {"id": "w0", "frame_range": [0, 2]})
        self.assertEqual(context["instruction"], "Original instruction")
        self.assertEqual(context["checklist"], matching_observation()["candidate_checks"])
        self.assertEqual(set(context["expert_catalog"]), {"object_spatial", "short_window_motion"})
        self.assertTrue(all(e["execution_status"] == "not_implemented" for e in context["expert_catalog"].values()))
        context["expert_catalog"].clear()
        self.assertEqual(len(EXPERT_CATALOG), 2)

    def test_related_checks_without_tool_coverage_are_retained(self):
        value = window_matching()
        value["checks"][0]["experts"] = []
        value["checks"][0]["limitations"] = ["No supplied tool can establish physical containment."]
        self.assertEqual(self.parse(value), value)

    def test_uncertain_or_uncovered_checks_need_an_explicit_limitation(self):
        for relevance, experts in (("uncertain", window_matching()["checks"][0]["experts"]), ("related", [])):
            with self.subTest(relevance=relevance):
                value = window_matching()
                value["checks"][0].update(relevance=relevance, experts=experts, limitations=[])
                with self.assertRaises(ValueError):
                    self.parse(value)
                value["checks"][0]["limitations"] = ["Visibility or tool coverage is insufficient."]
                self.assertEqual(self.parse(value), value)

    def test_missing_duplicate_and_invented_check_ids_are_rejected(self):
        for mode in ("missing", "duplicate", "invented"):
            with self.subTest(mode=mode):
                value = window_matching()
                if mode == "missing":
                    value["checks"].pop()
                elif mode == "duplicate":
                    value["checks"].append(copy.deepcopy(value["checks"][0]))
                else:
                    value["checks"][0]["check_id"] = "c999"
                with self.assertRaises(ValueError):
                    self.parse(value)

    def test_old_or_invented_expert_names_and_duplicate_calls_are_rejected(self):
        for expert_id in ("layout", "motion", "depth_oracle", "duplicate"):
            with self.subTest(expert_id=expert_id):
                value = window_matching()
                if expert_id == "duplicate":
                    value["checks"][0]["experts"].append(copy.deepcopy(value["checks"][0]["experts"][0]))
                else:
                    value["checks"][0]["experts"][0]["expert_id"] = expert_id
                with self.assertRaises(ValueError):
                    self.parse(value)

    def test_expert_entities_must_be_distinct_declared_check_entities(self):
        for entities in ([], ["apple", "apple"], ["invented_gripper"], ["drawer"]):
            with self.subTest(entities=entities):
                value = window_matching()
                value["checks"][0]["experts"][0]["entity_keys"] = entities
                with self.assertRaises(ValueError):
                    self.parse(value)

    def test_frames_must_be_supplied_and_motion_interval_must_be_continuous(self):
        for frames in ([], [3], [True, 2], [2, 1], [1, 1], [0, 2], [1]):
            with self.subTest(frames=frames):
                value = window_matching()
                value["checks"][0]["experts"][1]["frame_ids"] = frames
                with self.assertRaises(ValueError):
                    self.parse(value)

    def test_unrelated_checks_cannot_schedule_experts(self):
        value = window_matching()
        value["checks"][0]["relevance"] = "unrelated"
        with self.assertRaises(ValueError):
            self.parse(value)

    def test_wrong_window_prose_wrapped_json_or_completion_fields_are_rejected(self):
        value = window_matching()
        value["window_id"] = "w1"
        with self.assertRaises(ValueError):
            self.parse(value)
        with self.assertRaises(ValueError):
            parse_window_matching("```json\n" + json.dumps(window_matching()) + "\n```", {"id": "w0"},
                                  matching_observation()["candidate_checks"], [0, 1, 2])
        for location in ("top", "check", "expert"):
            with self.subTest(location=location):
                value = window_matching()
                target = value if location == "top" else value["checks"][0]
                if location == "expert":
                    target = target["experts"][0]
                target["verdict"] = "yes"
                with self.assertRaises(ValueError):
                    self.parse(value)

    def test_selected_windows_are_validated_against_all_original_decisions(self):
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p1"})
        self.assertEqual(validate_matching_inputs("Task", matching_observation(), selection, [i/10 for i in range(5)]),
                         selection["selected_windows"])
        for mode in ("unselected", "wrong_id", "missing_check"):
            with self.subTest(mode=mode):
                obs, changed = matching_observation(), copy.deepcopy(selection)
                if mode == "unselected":
                    changed["selected_windows"].append(changed["windows"][1])
                elif mode == "wrong_id":
                    changed["windows"][0]["id"] = "../unsafe"
                else:
                    obs["candidate_checks"].pop()
                with self.assertRaises(ValueError):
                    validate_matching_inputs("Task", obs, changed, [i/10 for i in range(5)])

    def test_missing_nonfinite_or_unordered_timestamps_are_rejected(self):
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p1"})
        for times in ([], [0, 1], [0, 1, float("nan"), 3, 4], [0, 2, 1, 3, 4], [0, True, 2, 3, 4]):
            with self.subTest(times=times):
                with self.assertRaises(ValueError):
                    validate_matching_inputs("Task", matching_observation(), selection, times)


if __name__ == "__main__":
    unittest.main()
