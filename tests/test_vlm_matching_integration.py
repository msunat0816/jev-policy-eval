"""Real JPEG/window extraction with mocked Qwen HTTP; no actual model/expert execution."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_vlm_matching import matching_observation, window_matching
from window_selection import summarize_selection

try:
    import evaluate_rollout as ev
except ModuleNotFoundError as exc:
    if exc.name not in ("cv2", "numpy"):
        raise
    ev = None


def qwen_response(value):
    return {"choices": [{"message": {"content": json.dumps(value)}}], "usage": {"total_tokens": 123}}


class FrameCapture:
    def __init__(self, count=7, fail_at=None):
        self.position = 0
        self.count = count
        self.fail_at = fail_at
        self.read_indices = []

    def set(self, _, index):
        self.position = index

    def read(self):
        index = self.position
        self.position += 1
        self.read_indices.append(index)
        if index == self.fail_at or index >= self.count:
            return False, None
        return True, ev.np.full((12, 16, 3), index * 20, dtype=ev.np.uint8)


@unittest.skipIf(ev is None, "OpenCV and NumPy required for video/image integration")
class VlmMatchingIntegrationTests(unittest.TestCase):
    def test_each_selected_window_gets_its_continuous_frames_full_checklist_and_model(self):
        obs = matching_observation()
        obs["frames"].append(dict(obs["frames"][-1], frame=6))
        selection = summarize_selection([(0, 2), (2, 4), (4, 6)], {"w0": "p2", "w1": "p1", "w2": "p3"})
        cap = FrameCapture()
        responses = [qwen_response(window_matching()), qwen_response(window_matching("w2", 4, 6))]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only", "DASHSCOPE_BASE_URL": "https://example.invalid/v1"}), \
                    patch.object(ev, "post", side_effect=responses) as post:
                result = ev.match_windows("Original instruction", obs, selection, [i/10 for i in range(7)], cap, folder)
            self.assertEqual(post.call_count, 2)
            self.assertEqual(cap.read_indices, [0, 1, 2, 4, 5, 6])
            for call, window_id, frame_ids in zip(post.call_args_list, ("w0", "w2"), ([0, 1, 2], [4, 5, 6])):
                url, _, body, _, name = call.args
                self.assertEqual(url, "https://example.invalid/v1/chat/completions")
                self.assertEqual((body["model"], body["enable_thinking"]), ("qwen3.8-flash", False))
                self.assertEqual(body["response_format"], {"type": "json_object"})
                parts = body["messages"][0]["content"]
                context = json.loads(parts[0]["text"].split("\nContext: ", 1)[1])
                self.assertEqual(context["instruction"], "Original instruction")
                self.assertEqual(context["checklist"], obs["candidate_checks"])
                self.assertEqual(context["window"]["id"], window_id)
                self.assertEqual(name, f"03-match-{window_id}")
                self.assertEqual(call.kwargs["image_directory"], f"matching_frames/{window_id}")
                self.assertEqual([part["text"] for part in parts[1:] if part["type"] == "text"],
                                 [f"frame {i}, PTS {i/10} seconds" for i in frame_ids])
                self.assertEqual(sum(part["type"] == "image_url" for part in parts), len(frame_ids))
                self.assertTrue(all((folder / f"matching_frames/{window_id}/{i}.jpg").exists() for i in frame_ids))
                self.assertTrue((folder / f"03-matching-{window_id}.json").exists())
            self.assertFalse((folder / "matching_frames/w1").exists())
            self.assertEqual([w["window_id"] for w in result["windows"]], ["w0", "w2"])
            self.assertEqual(result["status"], "proposals_only")
            self.assertEqual(json.loads((folder / "03-window-matching.json").read_text()), result)

    def test_empty_selection_saves_empty_matching_without_reads_or_api_calls(self):
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p0", "w1": "p1"})
        cap = FrameCapture()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ev, "post") as post:
                result = ev.match_windows("Task", matching_observation(), selection, [i/10 for i in range(5)], cap, Path(tmp))
            post.assert_not_called()
            self.assertEqual(cap.read_indices, [])
            self.assertEqual(result["windows"], [])
            self.assertTrue((Path(tmp) / "03-window-matching.json").exists())

    def test_decode_failure_or_invalid_selection_never_sends_matching_request(self):
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p1"})
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ev, "post") as post:
                with self.assertRaisesRegex(ValueError, "decode failed at frame 1"):
                    ev.match_windows("Task", matching_observation(), selection, [i/10 for i in range(5)],
                                     FrameCapture(fail_at=1), Path(tmp))
            post.assert_not_called()
            self.assertFalse((Path(tmp) / "03-window-matching.json").exists())
        selection["selected_windows"].append(selection["windows"][1])
        cap = FrameCapture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(ev, "post") as post:
            with self.assertRaises(ValueError):
                ev.match_windows("Task", matching_observation(), selection, [i/10 for i in range(5)], cap, Path(tmp))
            post.assert_not_called()
            self.assertEqual(cap.read_indices, [])

    def test_invalid_response_preserves_actual_http_trace_and_correct_image_directory(self):
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p1"})
        response = qwen_response({"window_id": "w0", "checks": []})
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only", "DASHSCOPE_BASE_URL": "https://example.invalid/v1"}), \
                    patch.object(ev, "urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as http:
                with self.assertRaises(ValueError):
                    ev.match_windows("Task", matching_observation(), selection, [i/10 for i in range(5)], FrameCapture(), folder)
            http.assert_called_once()
            trace = json.loads((folder / "03-match-w0.json").read_text())
            self.assertEqual(trace["response"], response)
            self.assertEqual(trace["usage"], {"total_tokens": 123})
            images = [p for p in trace["request"]["messages"][0]["content"] if p["type"] == "image_url"]
            self.assertEqual([p["image_url"] for p in images], ["JPEGs in matching_frames/w0/, in prompt order"] * 3)
            self.assertFalse((folder / "03-matching-w0.json").exists())
            self.assertFalse((folder / "03-window-matching.json").exists())

    def test_single_frame_window_can_request_spatial_evidence_without_motion(self):
        obs = matching_observation()
        obs["frames"] = obs["frames"][:1]
        for check in obs["candidate_checks"]:
            check["related_frames"] = [0]
        selection = summarize_selection([(0, 0)], {"w0": "p2"})
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only"}), \
                patch.object(ev, "post", return_value=qwen_response(window_matching("w0", 0, 0, motion=False))):
            cap = FrameCapture(count=1)
            result = ev.match_windows("Task", obs, selection, [0.0], cap, Path(tmp))
            self.assertEqual(cap.read_indices, [0])
            self.assertEqual(result["windows"][0]["frame_ids"], [0])


if __name__ == "__main__":
    unittest.main()
