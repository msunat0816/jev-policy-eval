"""Regression checks for retained legacy collection/scoring helpers, outside the active entry point."""
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_coarse_observation import observation
from test_measurement_adapter import LAYOUT_TASK, requirement

try:
    import cv2
    import numpy as np
    import evaluate_rollout as ev
except ModuleNotFoundError as exc:
    if exc.name not in ("cv2", "numpy"):
        raise
    ev = None


def routes(n_windows, n_checks, priority="p2", layout="useful", motion="useful"):
    return {f"w{i}c{j}_{part}": answer for i in range(n_windows) for j in range(n_checks)
            for part, answer in (("priority", priority), ("layout", layout), ("motion", motion))}


def explicit_observation():
    obs = observation()
    obs["requirements"] = [requirement()]
    check = obs["candidate_checks"][0]
    check.update(entity_keys=["red_block", "blue_target"], related_frames=[0, 4],
                 question="Are the image-plane bounding boxes entirely separated left to right at episode end?",
                 rationale="The sampled boxes are separated; verify the final image-plane relationship.")
    boxes = {"red_block": [5/64*1000, 6/48*1000, 15/64*1000, 16/48*1000],
             "blue_target": [35/64*1000, 6/48*1000, 45/64*1000, 16/48*1000]}
    obs["frames"] = [{"frame": f, "visible_facts": [], "uncertainties": [], "boxes": boxes}
                     for f in (0, 2, 4)]
    return obs


class SyntheticCapture:
    def __init__(self):
        rng = random.Random(14)
        patches = [np.array([rng.randrange(256) for _ in range(300)], dtype=np.uint8).reshape(10, 10, 3)
                   for _ in range(2)]
        self.frames = []
        self.position = 0
        for i in range(5):
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[6:16, 5+i:15+i] = patches[0]
            image[6:16, 35:45] = patches[1]
            self.frames.append(image)

    def set(self, _, value):
        self.position = value

    def read(self):
        frame = self.frames[self.position].copy()
        self.position += 1
        return True, frame


@unittest.skipIf(ev is None, "OpenCV and NumPy required")
class LegacyCollectionIntegrationTests(unittest.TestCase):
    def judge(self, task, obs, windows, evidence, answer="level_5"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "03-evidence.json"
            ev.save(path, {"task": task, "requirements": obs["requirements"],
                           "windows": windows, "evidence": evidence})
            with patch.object(ev, "ask_jev", return_value={"grade": answer}) as ask:
                result = ev.score_saved(path)
            return result, ask.call_args.args[0]

    def test_thresholds_max_priority_and_unavailable_experts_remain_explicit(self):
        obs = explicit_observation()
        obs["candidate_checks"].append(dict(obs["candidate_checks"][0], id="c1"))
        answers = routes(2, 2, priority="p0")
        answers.update(w0c1_priority="p3", w0c1_layout="not_applicable", w0c1_motion="insufficient_context",
                       w1c0_priority="p1", w1c1_priority="p2", w1c1_layout="not_applicable",
                       w1c1_motion="not_applicable")
        with patch.object(ev, "track_pair") as track:
            windows, evidence = ev.collect(None, LAYOUT_TASK, obs, [(0, 2), (2, 4)], answers,
                                           [i/10 for i in range(5)], (64, 48))
        track.assert_not_called()
        self.assertEqual([w["priority"] for w in windows], [3, 2])
        self.assertTrue(all(w["over_threshold"] for w in windows))
        self.assertEqual({e["pair_id"] for e in evidence}, {"w0c1", "w1c1"})
        self.assertEqual({e["reason"] for e in evidence}, {"expert_not_applicable", "expert_applicability_unknown"})

    def test_real_cpu_tracking_reused_across_checks_and_experts_and_judge_reads_saved_evidence(self):
        obs = explicit_observation()
        obs["candidate_checks"].append(dict(obs["candidate_checks"][0], id="c1"))
        with patch.object(ev, "track_pair", wraps=ev.track_pair) as track:
            windows, evidence = ev.collect(SyntheticCapture(), LAYOUT_TASK, obs, [(0, 4)], routes(1, 2),
                                           [i/10 for i in range(5)], (64, 48))
        self.assertEqual(track.call_count, 1)
        layout = evidence[0]
        motion = evidence[1]
        self.assertEqual(layout["measurement"]["end_box_gaps_px"]["image_left"], 16)
        self.assertEqual(layout["requirement_status"], "supported")
        self.assertAlmostEqual(motion["measurement"]["max_relative_excursion_px"], 4)
        self.assertEqual(motion["requirement_status"], "unknown")
        result, state = self.judge(LAYOUT_TASK, obs, windows, evidence)
        self.assertEqual(result["score"], 5)  # Synthetic fixture + mocked judge, not a real delivered score.
        self.assertTrue(result["supporting_evidence_ids"])
        self.assertNotIn("observations", state)
        for record in state["evidence"]:
            self.assertTrue({"priority", "applicability", "tracks"}.isdisjoint(record))

    def test_zero_checks_or_unselected_terminal_window_cannot_reuse_early_measurement_as_success(self):
        obs = explicit_observation()
        answers = routes(2, 1)
        answers["w1c0_priority"] = "p1"
        windows, evidence = ev.collect(SyntheticCapture(), LAYOUT_TASK, obs, [(0, 2), (2, 4)], answers,
                                       [i/10 for i in range(5)], (64, 48))
        self.assertTrue(all(e["requirement_status"] == "unknown" for e in evidence))
        result, _ = self.judge(LAYOUT_TASK, obs, windows, evidence)
        self.assertEqual(result["guard_reason"], "no_decisive_program_evidence")
        obs["candidate_checks"] = []
        windows, evidence = ev.collect(None, LAYOUT_TASK, obs, [(0, 4)], {}, [i/10 for i in range(5)], (64, 48))
        self.assertIsNone(windows[0]["priority"])
        self.assertEqual(evidence, [])
        self.assertIsNone(self.judge(LAYOUT_TASK, obs, windows, evidence)[0]["score"])

    def test_missing_seed_ambiguous_entities_lost_or_partial_tracks_never_give_requirement_verdict(self):
        for failure in ("missing_seed", "third_entity", "lost", "partial"):
            with self.subTest(failure=failure):
                obs, cap = explicit_observation(), SyntheticCapture()
                if failure == "missing_seed":
                    obs["frames"][0]["boxes"] = {}
                if failure == "third_entity":
                    obs["candidate_checks"][0]["entity_keys"].append("gripper")
                if failure == "lost":
                    cap.frames[3][6:16, 8:18] = 0
                tracker = patch.object(ev, "track_pair", return_value=([], None)) if failure == "partial" else patch.object(ev, "track_pair", wraps=ev.track_pair)
                with tracker:
                    windows, evidence = ev.collect(cap, LAYOUT_TASK, obs, [(0, 4)], routes(1, 1),
                                                   [i/10 for i in range(5)], (64, 48))
                self.assertTrue(all(e["requirement_status"] == "unknown" for e in evidence))
                self.assertTrue(all("reason" in e for e in evidence))
                self.assertIsNone(self.judge(LAYOUT_TASK, obs, windows, evidence)[0]["score"])

    def test_latest_opposite_layout_blocks_full_completion(self):
        obs = explicit_observation()
        rows = [{"frame": i, "subject_xyxy": [45, 6, 55, 16],
                 "reference_xyxy": [25, 6, 35, 16]} for i in range(5)]
        with patch.object(ev, "track_pair", return_value=(rows, None)):
            windows, evidence = ev.collect(None, LAYOUT_TASK, obs, [(0, 4)], routes(1, 1),
                                           [i/10 for i in range(5)], (64, 48))
        self.assertEqual(evidence[0]["requirement_status"], "contradicted")
        result, _ = self.judge(LAYOUT_TASK, obs, windows, evidence)
        self.assertIsNone(result["score"])
        self.assertEqual(result["guard_reason"], "full_completion_lacks_evidence_for_each_requirement")


if __name__ == "__main__":
    unittest.main()
