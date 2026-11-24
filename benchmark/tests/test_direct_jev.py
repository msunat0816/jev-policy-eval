"""Regression checks for the whole-episode sampling and one-call contract."""
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmark import direct_jev, runner, worker
from benchmark.common import read_json, write_json


@unittest.skipUnless(shutil.which("ffprobe"), "ffprobe required")
class DirectJevTests(unittest.TestCase):
    def setUp(self):
        import cv2
        self.cv2 = cv2
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def video(self, count):
        import numpy as np
        path = self.folder / "neutral.mp4"
        video = self.cv2.VideoWriter(str(path), self.cv2.VideoWriter_fourcc(*"mp4v"),
                                     10, (320, 240))
        self.assertTrue(video.isOpened())
        try:
            for index in range(count):
                video.write(np.full((240, 320, 3), index, dtype=np.uint8))
        finally:
            video.release()
        return path

    def test_short_video_uses_both_endpoints_once(self):
        import evaluate_rollout as ev
        for count, expected in ((6, [0, 5]), (1, [0])):
            with self.subTest(count=count):
                path = self.video(count)
                out = self.folder / str(count)
                out.mkdir()
                body, manifest = direct_jev.prepare_request(ev, {"video": str(path), "task": "Move the cup"}, out)
                self.assertEqual([frame["frame_id"] for frame in body["state"]["frames"]], expected)
                self.assertEqual(len(manifest), len(expected))
                self.assertEqual(next(iter(manifest.values()))["prepared_size_hw"], [280, 320])

    def test_worker_sends_one_label_free_request_and_retains_usage(self):
        import evaluate_rollout as ev
        task = "Put the blue cup next to the plate."
        path = self.video(45)
        out = self.folder / "rollout"
        job = self.folder / "job.json"
        write_json(job, {"video": str(path), "task": task, "episode": "s1", "split": "validation",
                         "out": str(out), "sample_interval_seconds": 2.0, "max_calls": 1,
                         "timeout_seconds": 10.0})
        response = {"model": ev.FINAL_MODEL, "answers": {"grade": {"type": "choice", "choice": "level_4"}},
                    "usage": {"input_tokens": 400, "output_tokens": 2}}
        args = ["worker", "--job", str(job), "--framework-root", str(Path(ev.__file__).parent),
                "--method", direct_jev.METHOD]
        import signal
        original_alarm = signal.getsignal(signal.SIGALRM)
        try:
            with (patch.object(sys, "argv", args), patch.object(sys, "path", sys.path.copy()),
                  patch.object(ev, "post", ev.post),
                  patch.dict(os.environ, {"TYPESAFE_API_KEY": "fixture-only"}),
                  patch.object(ev, "urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as request):
                worker.main()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, original_alarm)
        request.assert_called_once()
        body = json.loads(request.call_args.args[0].data)
        self.assertEqual(set(body["state"]), {"instruction", "rubric", "sample_interval_seconds", "episode", "frames"})
        self.assertEqual(body["state"]["instruction"], task)
        self.assertEqual(body["state"]["rubric"], ev.RUBRIC)
        self.assertEqual(body["questions"]["grade"]["criteria"], ev.FINAL_CRITERIA)
        frames = body["state"]["frames"]
        self.assertEqual([frame["frame_id"] for frame in frames], [0, 20, 40, 44])
        self.assertEqual({frame["image"] for frame in frames}, set(body["assets"]))
        self.assertEqual(read_json(out / "result.json")["score"], 4)
        artifacts = runner.collect_artifacts(self.folder)
        self.assertEqual(artifacts["sampling"]["image_count"], 4)
        call, = artifacts["calls"]
        self.assertEqual(call["usage"], response["usage"])
        self.assertTrue(call["transport_attempted"])
        self.assertEqual(call["request_bytes"], len(request.call_args.args[0].data))
        self.assertNotIn("fixture-only", (out / "evaluate.json").read_text())

    def test_frame_limit_preserves_planned_ids(self):
        import evaluate_rollout as ev
        with patch.object(direct_jev, "MAX_FRAMES", 1), self.assertRaisesRegex(ValueError, "context guard"):
            direct_jev.prepare_request(ev, {"video": str(self.video(6)), "task": "Move the cup"}, self.folder)
        self.assertEqual([f["frame_id"] for f in read_json(self.folder / "sampling.json")["frames"]], [0, 5])

    def test_duration_pilot_includes_extremes_and_is_deterministic(self):
        inputs = [{"sample_id": f"s{i}"} for i in reversed(range(50))]
        durations = {f"s{i}": {"duration_s": i + 1} for i in range(50)}
        ids = direct_jev.select_pilot(inputs, durations, 10)
        self.assertEqual((ids[0], ids[-1], len(set(ids))), ("s0", "s49", 10))
        self.assertEqual(ids, direct_jev.select_pilot(inputs[::-1], durations, 10))
