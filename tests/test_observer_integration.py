"""Synthetic image + mocked API checks; this does not validate any real model."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_coarse_observation import observation

try:
    import cv2
    import numpy as np
    import evaluate_rollout as ev
except ModuleNotFoundError as exc:
    if exc.name not in ("cv2", "numpy"):
        raise
    ev = None


@unittest.skipIf(ev is None, "OpenCV and NumPy required; installed by CPU CI")
class ObserverIntegrationTests(unittest.TestCase):
    def test_one_request_keeps_approved_prompt_model_ids_and_sample_labels(self):
        from coarse_observation import OBSERVE, PROMPT_PATH
        response = {"choices": [{"message": {"content": json.dumps(observation())}}]}
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        cap = unittest.mock.Mock()
        cap.read.return_value = (True, image)
        times = [i / 10 for i in range(28)]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only",
                                         "DASHSCOPE_BASE_URL": "https://example.invalid/compatible-mode/v1"}), \
                    patch.object(ev, "post", return_value=response) as post:
                result = ev.observe("Place the beige block on the left peg.", [7, 27], times, cap, folder)
            self.assertEqual(post.call_count, 1)
            url, _, body, _, name = post.call_args.args
            self.assertEqual(url, "https://example.invalid/compatible-mode/v1/chat/completions")
            self.assertEqual((body["model"], body["enable_thinking"]), ("qwen3.8-flash", False))
            self.assertEqual(OBSERVE, PROMPT_PATH.read_text(encoding="utf-8"))
            parts = body["messages"][0]["content"]
            self.assertTrue(parts[0]["text"].startswith(OBSERVE))
            self.assertEqual([p["text"] for p in parts[1:] if p["type"] == "text"],
                             ["frame 7, PTS 0.7 seconds", "frame 27, PTS 2.7 seconds"])
            images = [p for p in parts if p["type"] == "image_url"]
            self.assertEqual(len(images), 2)
            self.assertTrue(all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in images))
            self.assertEqual(name, "01-observe")
            self.assertEqual(json.loads((folder / "01-observation.json").read_text()), result)
            self.assertEqual(sorted(p.name for p in (folder / "frames").iterdir()), ["27.jpg", "7.jpg"])

    def test_new_requirements_keep_measurements_without_claiming_semantic_support(self):
        obs = observation()
        rows = [{"frame": i, "subject_xyxy": [5, 5, 10, 10],
                 "reference_xyxy": [25, 5, 30, 10]} for i in range(7, 28)]
        with patch.object(ev, "track_pair", return_value=(rows, None)):
            decisions, evidence = ev.collect(None, "Place the beige block on the left peg.", obs, [(7, 27)],
                                             {"w0c0_priority": "p2", "w0c0_layout": "useful",
                                              "w0c0_motion": "not_applicable"},
                                             [i / 10 for i in range(28)], (64, 48))
        self.assertEqual(evidence[0]["status"], "measurement_only")
        self.assertEqual(evidence[0]["requirement_status"], "unknown")
        self.assertEqual(evidence[0]["measurement_status"], "available")
        self.assertEqual(evidence[0]["semantic_unknown_reason"], "original_instruction_outside_supported_language")
        self.assertIn("end_box_gaps_px", evidence[0]["measurement"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "03-evidence.json"
            ev.save(path, {"task": "Place the beige block on the left peg.",
                           "requirements": obs["requirements"], "windows": decisions, "evidence": evidence})
            with patch.object(ev, "ask_jev", return_value={"grade": "level_5"}):
                result = ev.score_saved(path)
            self.assertIsNone(result["score"])
            self.assertEqual(result["guard_reason"], "no_decisive_program_evidence")


if __name__ == "__main__":
    unittest.main()
