"""Mocked JEV selection and entry-point boundary checks; no real model calls."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from test_window_selection import selection_observation
from test_vlm_matching import matching_observation, window_matching
from test_vlm_matching_integration import qwen_response

try:
    import evaluate_rollout as ev
except ModuleNotFoundError as exc:
    if exc.name not in ("cv2", "numpy"):
        raise
    ev = None


def response(answers):
    return {"answers": {key: {"choice": value} for key, value in answers.items()}}


@unittest.skipIf(ev is None, "OpenCV and NumPy required for the video entry point")
class WindowSelectionIntegrationTests(unittest.TestCase):
    def test_single_batched_request_contains_only_window_choices_and_persists_selection(self):
        obs = selection_observation()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), \
                    patch.object(ev, "post", return_value=response({"w0": "p2", "w1": "p1"})) as post:
                result = ev.select_windows("Task", obs, [(0, 2), (2, 4)], folder)
            post.assert_called_once()
            url, _, body, _, name = post.call_args.args
            self.assertEqual(url, "https://api.typesafe.ai/v1/systemone")
            self.assertEqual(body["model"], "jev-1.14.0")
            self.assertEqual(name, "02-select-windows")
            state = json.loads(body["state"])
            self.assertEqual(state["checklist"], obs["candidate_checks"])
            self.assertEqual(state["requirements"], obs["requirements"])
            self.assertEqual(set(body["questions"]), {"w0", "w1"})
            self.assertEqual(json.loads((folder / "02-selected-windows.json").read_text()), result)
            self.assertEqual([w["id"] for w in result["selected_windows"]], ["w0"])

    def test_invalid_checklist_or_model_output_cannot_create_selection_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            obs = selection_observation()
            obs["candidate_checks"] = []
            with patch.object(ev, "ask_jev") as ask:
                with self.assertRaises(ValueError):
                    ev.select_windows("Task", obs, [(0, 2)], folder)
            ask.assert_not_called()
            with patch.object(ev, "ask_jev", return_value={"w0c0_priority": "p2"}):
                with self.assertRaises(ValueError):
                    ev.select_windows("Task", selection_observation(), [(0, 2)], folder)
            self.assertFalse((folder / "02-selected-windows.json").exists())

    def test_empty_candidates_are_saved_without_a_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with patch.object(ev, "ask_jev") as ask:
                result = ev.select_windows("Task", selection_observation(), [], folder)
            ask.assert_not_called()
            self.assertEqual(result["selected_windows"], [])
            self.assertEqual(json.loads((folder / "02-selected-windows.json").read_text()), result)

    def test_main_stops_after_matching_even_when_no_window_is_selected(self):
        for answers, selected in (({"w0": "p2", "w1": "p1"}, ["w0"]),
                                  ({"w0": "p0", "w1": "p1"}, [])):
            with self.subTest(answers=answers), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                video = root / "synthetic-video.mp4"
                video.write_bytes(b"Synthetic fixture: decoding is mocked.")
                args = SimpleNamespace(video=video, out=root / "run", task="Task", split="test", episode="synthetic")
                cap = Mock()
                cap.get.side_effect = [64, 48]
                cap.read.return_value = (True, ev.np.zeros((12, 16, 3), dtype=ev.np.uint8))
                responses = [response(answers)] + ([qwen_response(window_matching())] if selected else [])
                with patch.object(ev, "frame_times", return_value=[i/10 for i in range(5)]), \
                        patch.object(ev.cv2, "VideoCapture", return_value=cap), \
                        patch.object(ev, "observe", return_value=matching_observation()) as observe, \
                        patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only", "DASHSCOPE_API_KEY": "test-only"}), \
                        patch.object(ev, "post", side_effect=responses) as post, \
                        patch.object(ev, "collect") as collect, \
                        patch.object(ev, "score_saved") as score, \
                        patch.object(ev, "track_pair") as track:
                    result = ev.run(args, sample_interval_seconds=0.2)
                observe.assert_called_once()
                self.assertEqual(observe.call_args.args[1], [0, 2, 4])
                self.assertEqual(post.call_count, 2 if selected else 1)
                collect.assert_not_called()
                score.assert_not_called()
                track.assert_not_called()
                cap.release.assert_called_once()
                self.assertEqual(result["status"], "awaiting_expert_execution")
                self.assertIsNone(result["score"])
                self.assertEqual([w["id"] for w in result["selected_windows"]], selected)
                self.assertEqual(json.loads((args.out / "result.json").read_text()), result)
                self.assertTrue((args.out / result["selection_file"]).exists())
                self.assertTrue((args.out / result["matching_file"]).exists())
                self.assertEqual(result["matched_window_count"], len(selected))
                self.assertFalse((args.out / "03-evidence.json").exists())
                metadata = json.loads((args.out / "run.json").read_text())
                self.assertEqual(metadata["window_selection"]["unit"], "window")
                self.assertEqual(metadata["window_selection"]["choices_per_window"], ["priority"])
                self.assertNotIn("routing", metadata)
                self.assertNotIn("expert_config", metadata)
                self.assertEqual(metadata["vlm_matching"]["model"], "qwen3.8-flash")
                self.assertFalse(metadata["vlm_matching"]["expert_execution"])


if __name__ == "__main__":
    unittest.main()
