"""Pure selection contracts, including checklist coverage and per-window decisions."""
import copy
import unittest

from test_coarse_observation import observation
from window_selection import (SELECTION_PROTOCOL, WINDOW_THRESHOLD, build_selection_request,
                              summarize_selection)


def selection_observation():
    obs = observation()
    obs["requirements"].append({"id": "r1", "text": "Then close the drawer.",
                                "subject": "drawer", "reference": ""})
    obs["frames"] = [
        {"frame": frame, "visible_facts": [fact], "uncertainties": unknowns, "boxes": {}}
        for frame, fact, unknowns in (
            (0, "The block is beside the peg.", []),
            (2, "The block appears to overlap the peg.", ["Support is unclear."]),
            (4, "The drawer front is visible.", ["The block is obscured."]),
        )
    ]
    obs["candidate_checks"][0].update(related_frames=[2, 4])
    obs["candidate_checks"].append({
        "id": "c1", "requirement_ids": ["r1"], "entity_keys": ["drawer"],
        "related_frames": [4], "question": "Is the drawer closed at episode end?",
        "rationale": "Its front is visible, but the closed state still needs evaluation.",
    })
    return obs


class WindowSelectionTests(unittest.TestCase):
    def test_full_checklist_and_all_descriptions_are_shared_with_one_choice_per_window(self):
        obs = selection_observation()
        original = copy.deepcopy(obs)
        state, questions = build_selection_request("Original instruction", obs, [(0, 2), (2, 4)])
        self.assertEqual(state["task"], "Original instruction")
        self.assertEqual(state["requirements"], obs["requirements"])
        self.assertEqual(state["checklist"], obs["candidate_checks"])
        self.assertEqual(len(state["coarse_observations"]), 3)
        self.assertEqual(state["coarse_observations"][1]["uncertainties"], ["Support is unclear."])
        self.assertTrue(all("boxes" not in frame for frame in state["coarse_observations"]))
        self.assertEqual(set(questions), {"w0", "w1"})
        for question in questions.values():
            self.assertEqual(question["type"], "choice")
            self.assertEqual(set(question["criteria"]), {"p0", "p1", "p2", "p3"})
        self.assertEqual(obs, original)

    def test_checks_citing_only_later_frames_do_not_filter_candidate_windows(self):
        obs = selection_observation()
        for check in obs["candidate_checks"]:
            check["related_frames"] = [4]
        state, questions = build_selection_request("Task", obs, [(0, 2), (2, 4)])
        self.assertEqual(state["candidate_windows"], [
            {"id": "w0", "frame_range": [0, 2]}, {"id": "w1", "frame_range": [2, 4]},
        ])
        self.assertEqual(len(questions), 2)
        self.assertIn('"frame_range": [0, 2]', questions["w0"]["instructions"])

    def test_more_checks_do_not_add_model_choices_or_preassign_a_check(self):
        obs = selection_observation()
        obs["candidate_checks"].append(dict(obs["candidate_checks"][0], id="c2"))
        state, questions = build_selection_request("Task", obs, [(0, 2), (2, 4)])
        self.assertEqual(len(state["checklist"]), 3)
        self.assertEqual(len(questions), 2)
        self.assertTrue(all("candidate_check" not in question for question in questions.values()))

    def test_threshold_all_decisions_and_original_ids_and_order_are_preserved(self):
        windows = [(0, 2), (2, 4), (4, 6), (6, 8)]
        result = summarize_selection(windows, {"w3": "p3", "w1": "p1", "w2": "p2", "w0": "p0"})
        self.assertEqual(result["protocol"], SELECTION_PROTOCOL)
        self.assertEqual(result["threshold"], WINDOW_THRESHOLD)
        self.assertEqual([w["priority"] for w in result["windows"]], [0, 1, 2, 3])
        self.assertEqual([w["selected"] for w in result["windows"]], [False, False, True, True])
        self.assertEqual([w["id"] for w in result["selected_windows"]], ["w2", "w3"])
        self.assertEqual([w["frame_range"] for w in result["selected_windows"]], [[4, 6], [6, 8]])

    def test_no_quota_and_terminal_window_uses_same_rule(self):
        windows = [(0, 2), (2, 4)]
        for answers, expected in (({"w0": "p0", "w1": "p1"}, []),
                                  ({"w0": "p2", "w1": "p1"}, ["w0"]),
                                  ({"w0": "p2", "w1": "p2"}, ["w0", "w1"])):
            with self.subTest(answers=answers):
                result = summarize_selection(windows, answers)
                self.assertEqual([w["id"] for w in result["selected_windows"]], expected)

    def test_empty_or_structurally_incomplete_checklist_is_rejected(self):
        for checks in ([], selection_observation()["candidate_checks"][:1]):
            with self.subTest(checks=checks):
                obs = selection_observation()
                obs["candidate_checks"] = checks
                with self.assertRaisesRegex(ValueError, "complete checklist required"):
                    build_selection_request("Task", obs, [(0, 2), (2, 4)])

    def test_missing_extra_invalid_or_old_pair_answers_are_rejected(self):
        for answers in ({}, {"w0": "p2", "w1": "p2"}, {"w0": "useful"}, {"w0": 2},
                        {"w0": []}, {"w0c0_priority": "p2"}):
            with self.subTest(answers=answers):
                with self.assertRaises(ValueError):
                    summarize_selection([(0, 2)], answers)

    def test_invalid_or_unobserved_window_bounds_are_rejected(self):
        for windows in ([(0, 2), (0, 2)], [(2, 0)], [(True, 2)], [(-1, 2)], [(0, 3)], [(0,)]):
            with self.subTest(windows=windows):
                with self.assertRaises(ValueError):
                    build_selection_request("Task", selection_observation(), windows)

    def test_single_frame_and_empty_candidates_are_supported(self):
        _, questions = build_selection_request("Task", selection_observation(), [(0, 0)])
        self.assertEqual(set(questions), {"w0"})
        self.assertEqual(summarize_selection([(0, 0)], {"w0": "p2"})["selected_windows"][0]["frame_range"], [0, 0])
        _, questions = build_selection_request("Task", selection_observation(), [])
        self.assertEqual(questions, {})
        self.assertEqual(summarize_selection([], {})["selected_windows"], [])


if __name__ == "__main__":
    unittest.main()
