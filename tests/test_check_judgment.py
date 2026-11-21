"""Actual JPEG/HTTP serialization with mocked JEV; no paid service calls."""
import base64
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import test_evidence_packaging as packaging_tests
import test_expert_execution as execution_tests
from test_expert_execution import saved_matching
from test_vlm_matching import candidate_bundle, matching_observation, window_matching
from test_vlm_matching_integration import ev
from window_selection import summarize_selection

if ev is not None:
    import check_judgment as cj


def jev_response(check_id="c0", answer="yes"):
    return {"model": "jev-1.14.0", "answers": {check_id: {
        "type": "choice", "choice": answer, "confidence": .15,
        "probabilities": {key: .34 if key == answer else .33 for key in ("yes", "no", "unknown")}}},
        "usage": {"input_tokens": 100, "output_tokens": 10}}


@unittest.skipIf(ev is None, "OpenCV and NumPy required")
class CheckJudgmentTests(unittest.TestCase):
    def fixture(self, folder, *, both_related=False, empty_masks=False):
        helper = packaging_tests.EvidencePackagingTests()
        inputs, execution = helper.inputs(empty_masks=empty_masks)
        if both_related:
            inputs[-1]["windows"][0]["checks"][1].update(
                relevance="related", reason="The images may show the drawer.", limitations=["No drawer expert measurement."])
            execution = execution_tests.ExpertExecutionTests().run_execution(inputs)
        packages = helper.build(inputs, execution)
        self.write_images(folder, inputs)
        return inputs, execution, packages

    def write_images(self, folder, inputs):
        by_frame = {f["frame_id"]: f for f in inputs[3]["frames"]}
        for window in inputs[-1]["windows"]:
            for frame in window["images"]:
                image = ev.np.full((*by_frame[frame["frame_id"]]["image_size_hw"], 3), frame["frame_id"] * 20, ev.np.uint8)
                for key, content in (("original_file", image), ("candidates_file", ev.render_candidates(image, by_frame[frame["frame_id"]]))):
                    path = folder / frame[key]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.assertTrue(ev.cv2.imwrite(str(path), content))

    def judge(self, folder, inputs, execution, packages):
        obs, selection, times, candidates, matching = inputs
        return ev.judge_checks("Task", obs, selection, times, folder, candidates, matching, execution, packages)

    def test_actual_http_has_image_header_assets_one_choice_and_exact_input_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            inputs, execution, packages = self.fixture(folder)
            before = copy.deepcopy((inputs, execution, packages))
            sent = []
            def http(request, timeout):
                sent.append(request)
                return io.BytesIO(json.dumps(jev_response()).encode())
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch.object(ev, "urlopen", side_effect=http) as urlopen:
                result = self.judge(folder, inputs, execution, packages)
            urlopen.assert_called_once()
            request, = sent
            self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
            self.assertEqual(dict((k.lower(), v) for k, v in request.header_items())["typesafe-danger-zone-images"], "true")
            body = json.loads(request.data)
            self.assertEqual(body["model"], "jev-1.14.0")
            self.assertEqual(set(body["questions"]), {"c0"})
            self.assertEqual(set(body["questions"]["c0"]["criteria"]), {"yes", "no", "unknown"})
            state = body["state"]
            self.assertEqual(state["instruction"], "Task")
            self.assertEqual(state["current_check"], inputs[0]["candidate_checks"][0])
            self.assertEqual(state["requirements"], packages["packages"][0]["requirements"])
            self.assertNotIn("checklist", state)
            frames = state["evidence"]["frames"]
            self.assertEqual([f["frame_id"] for f in frames], [0, 1, 2])
            self.assertEqual([f["pts_seconds"] for f in frames], [0, .09, .25])
            self.assertEqual(len(body["assets"]), 6)
            reference_text = json.dumps({"state": body["state"], "questions": body["questions"]})
            for reference, asset in body["assets"].items():
                self.assertEqual(reference_text.count(reference), 1)
                image = ev.cv2.imdecode(ev.np.frombuffer(base64.b64decode(asset["data_base64"], validate=True), ev.np.uint8), ev.cv2.IMREAD_COLOR)
                self.assertTrue(all(280 <= dim <= 1568 for dim in image.shape[:2]))
            self.assertEqual((inputs, execution, packages), before)
            self.assertEqual([r["answer"] for r in result["judgments"]], ["yes", "unknown"])
            self.assertEqual([r["answer_source"] for r in result["judgments"]], ["jev", "program_no_evidence"])
            self.assertEqual(result["jev_call_count"], 1)
            self.assertEqual(result["judgments"][0]["confidence"], .15)  # No confidence cutoff changes yes.
            trace = json.loads((folder / "06-judge-c0000.json").read_text())
            self.assertEqual(trace["response"], jev_response())
            self.assertEqual(trace["headers"], cj.IMAGE_HEADERS)
            self.assertNotIn("test-only", json.dumps(trace))
            for reference, asset in body["assets"].items():
                self.assertNotEqual(trace["request"]["assets"][reference]["data_base64"], asset["data_base64"])
                metadata = trace["image_assets"][reference]
                self.assertEqual((folder / metadata["prepared_file"]).read_bytes(), base64.b64decode(asset["data_base64"]))
            self.assertTrue((folder / "06-check-c0000.json").exists())
            self.assertTrue((folder / "06-check-c0001.json").exists())
            self.assertEqual(json.loads((folder / "06-check-judgments.json").read_text()), result)
            self.assertNotIn("supporting_evidence_ids", result["judgments"][0])

    def test_two_checks_get_separate_contexts_and_each_one_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            inputs, execution, packages = self.fixture(folder, both_related=True)
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), \
                    patch.object(ev, "post", side_effect=[jev_response("c0", "no"), jev_response("c1", "unknown")]) as post:
                result = self.judge(folder, inputs, execution, packages)
            self.assertEqual(post.call_count, 2)
            self.assertEqual(result["jev_call_count"], 2)
            for i, call in enumerate(post.call_args_list):
                body = call.args[2]
                self.assertEqual(set(body["questions"]), {f"c{i}"})
                self.assertEqual(body["state"]["current_check"]["id"], f"c{i}")
            self.assertEqual([r["answer"] for r in result["judgments"]], ["no", "unknown"])
            self.assertEqual(post.call_args_list[1].args[2]["state"]["evidence"]["measurements"], [])

    def test_all_related_windows_are_combined_once_without_window_voting(self):
        obs = matching_observation()
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p3"})
        times = [0, .09, .25, .4, .6]
        values = [window_matching("w0", 0, 2), window_matching("w1", 2, 4)]
        values[1]["checks"][0].update(relevance="uncertain", limitations=["Identity uncertain."])
        inputs = obs, selection, times, candidate_bundle(selection, times), saved_matching("Task", obs, selection, times, values)
        execution = execution_tests.ExpertExecutionTests().run_execution(inputs)
        packages = packaging_tests.EvidencePackagingTests().build(inputs, execution)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write_images(folder, inputs)
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch.object(ev, "post", return_value=jev_response()) as post:
                self.judge(folder, inputs, execution, packages)
            post.assert_called_once()
            evidence = post.call_args.args[2]["state"]["evidence"]
            self.assertEqual([w["window_id"] for w in evidence["windows"]], ["w0", "w1"])
            self.assertEqual(evidence["windows"][1]["relevance"], "uncertain")
            self.assertEqual([f["frame_id"] for f in evidence["frames"]], [0, 1, 2, 3, 4])
            self.assertEqual(len(post.call_args.args[2]["assets"]), 10)
            self.assertEqual(len(evidence["frames"][2]["sources"]), 2)

    def test_missing_experts_do_not_prevent_image_based_yes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            inputs, execution, packages = self.fixture(folder, empty_masks=True)
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch.object(ev, "post", return_value=jev_response()) as post:
                result = self.judge(folder, inputs, execution, packages)
            post.assert_called_once()
            self.assertEqual(result["judgments"][0]["answer"], "yes")
            evidence = post.call_args.args[2]["state"]["evidence"]
            self.assertEqual(evidence["measurements"][1]["execution_status"], "not_executed")

    def test_empty_selection_is_program_unknown_with_no_images_or_api(self):
        obs = matching_observation()
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p0", "w1": "p1"})
        times = [i/10 for i in range(5)]
        inputs = obs, selection, times, candidate_bundle(selection, times), saved_matching("Task", obs, selection, times)
        execution = execution_tests.ExpertExecutionTests().run_execution(inputs)
        packages = packaging_tests.EvidencePackagingTests().build(inputs, execution)
        with tempfile.TemporaryDirectory() as tmp, patch.object(ev, "post") as post:
            folder = Path(tmp)
            result = self.judge(folder, inputs, execution, packages)
            self.assertFalse((folder / "check_frames").exists())
        post.assert_not_called()
        self.assertEqual(result["jev_call_count"], 0)
        self.assertTrue(all(r["answer"] == "unknown" and r["answer_source"] == "program_no_evidence"
                            and r["actual_model"] is None and r["request_file"] is None for r in result["judgments"]))

    def test_preparation_preserves_full_canvas_and_source_and_records_transforms(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for shape in ((100, 200), (800, 2400), (280, 1568), (1568, 50), (1568, 1568)):
                name = f"{shape[0]}x{shape[1]}"
                source = folder / f"{name}.jpg"
                original = ev.np.full((*shape, 3), 60, ev.np.uint8)
                original[:10, :10] = 220
                self.assertTrue(ev.cv2.imwrite(str(source), original))
                before = source.read_bytes()
                data, metadata = cj.prepare_image(folder, source.name, f"{name}-prepared.jpg", kind="original")
                prepared = ev.cv2.imdecode(ev.np.frombuffer(data, ev.np.uint8), ev.cv2.IMREAD_COLOR)
                self.assertEqual(source.read_bytes(), before)
                self.assertEqual(metadata["source_size_hw"], list(shape))
                self.assertTrue(all(280 <= d <= 1568 for d in prepared.shape[:2]))
                sx, sy = metadata["resize_scale_xy"]
                left, top, right, bottom = metadata["padding_ltrb_px"]
                self.assertEqual(round(shape[1] * sx) + left + right, prepared.shape[1])
                self.assertEqual(round(shape[0] * sy) + top + bottom, prepared.shape[0])
                self.assertEqual((folder / metadata["prepared_file"]).read_bytes(), data)
                if shape in ((280, 1568), (1568, 1568)):
                    self.assertEqual(data, before)

    def test_response_choices_are_structured_and_metadata_never_changes_choice(self):
        for answer in ("yes", "no", "unknown"):
            result = cj.parse_check_response(jev_response(answer=answer), "c0")
            self.assertEqual(result["answer"], answer)
            self.assertNotIn("explanation", result)
        response = jev_response()
        response["answers"]["c0"].update(confidence=0, probabilities={"yes": .1, "no": .1, "unknown": .8})
        self.assertEqual(cj.parse_check_response(response, "c0")["answer"], "yes")
        del response["usage"]
        self.assertIsNone(cj.parse_check_response(response, "c0")["usage"])

    def test_invalid_response_shapes_and_values_are_rejected(self):
        for mode in ("missing", "extra", "type", "choice", "model", "confidence", "probabilities"):
            with self.subTest(mode=mode):
                value = jev_response()
                if mode == "missing": value["answers"].clear()
                elif mode == "extra": value["answers"]["invented"] = value["answers"]["c0"]
                elif mode == "type": value["answers"]["c0"]["type"] = "noul"
                elif mode == "choice": value["answers"]["c0"]["choice"] = "supported"
                elif mode == "model": value["model"] = "other-model"
                elif mode == "confidence": value["answers"]["c0"]["confidence"] = float("nan")
                else: value["answers"]["c0"]["probabilities"]["yes"] = True
                with self.assertRaises(ValueError): cj.parse_check_response(value, "c0")

    def test_corrupt_evidence_boundary_fails_before_images_or_api_calls(self):
        for mode in ("check", "summary", "frame", "duplicate", "boolean"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                inputs, execution, packages = self.fixture(folder)
                packet = packages["packages"][0]
                if mode == "check": packet["check_id"] = "invented"
                elif mode == "summary": packet["measurements"][0]["summary"]["pairs"][0]["metrics"]["box_iou"]["max"]["value"] = .9
                elif mode == "frame": packet["images"][0]["pts_seconds"] = 999
                elif mode == "duplicate": packages["packages"].append(copy.deepcopy(packet))
                else: packet["measurements"][0]["summary"]["pairs"][0]["metrics"]["box_iou"]["max"]["value"] = False
                with patch.object(ev, "post") as post:
                    with self.assertRaisesRegex(ValueError, "upstream records"):
                        self.judge(folder, inputs, execution, packages)
                post.assert_not_called()
                self.assertFalse((folder / "check_frames").exists())
                self.assertFalse((folder / "06-check-judgments.json").exists())

    def test_missing_or_corrupt_image_is_a_failure_not_program_unknown(self):
        for mode in ("missing", "corrupt"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                inputs, execution, packages = self.fixture(folder)
                source = folder / packages["packages"][0]["images"][0]["original_file"]
                if mode == "missing": source.unlink()
                else: source.write_bytes(b"Not an image")
                with patch.object(ev, "post") as post:
                    with self.assertRaisesRegex(ValueError, "image"):
                        self.judge(folder, inputs, execution, packages)
                post.assert_not_called()
                failure = json.loads((folder / "06-failed-c0000.json").read_text())
                self.assertEqual(failure["error_type"], "ValueError")
                self.assertNotIn("answer", failure)
                self.assertFalse((folder / "06-check-c0000.json").exists())
                self.assertFalse((folder / "06-check-judgments.json").exists())

    def test_invalid_second_response_keeps_first_judgment_and_both_raw_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            inputs, execution, packages = self.fixture(folder, both_related=True)
            wrong = jev_response("c1")
            wrong["answers"]["c1"]["choice"] = "level_5"
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch.object(ev, "urlopen", side_effect=[
                    io.BytesIO(json.dumps(jev_response()).encode()), io.BytesIO(json.dumps(wrong).encode())]) as http:
                with self.assertRaisesRegex(ValueError, "yes/no/unknown"):
                    self.judge(folder, inputs, execution, packages)
            self.assertEqual(http.call_count, 2)
            self.assertEqual(json.loads((folder / "06-check-c0000.json").read_text())["answer"], "yes")
            self.assertEqual(json.loads((folder / "06-judge-c0001.json").read_text())["response"], wrong)
            self.assertTrue((folder / "06-failed-c0001.json").exists())
            self.assertFalse((folder / "06-check-c0001.json").exists())
            self.assertFalse((folder / "06-check-judgments.json").exists())

    def test_service_context_error_is_retained_without_retry_or_image_dropping(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            inputs, execution, packages = self.fixture(folder)
            error_body = {"detail": "max_tokens_exceeded"}
            error = HTTPError(cj.JEV_URL, 400, "Bad Request", None, io.BytesIO(json.dumps(error_body).encode()))
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch.object(ev, "urlopen", side_effect=error) as http:
                with self.assertRaises(HTTPError): self.judge(folder, inputs, execution, packages)
            http.assert_called_once()
            trace = json.loads((folder / "06-judge-c0000.json").read_text())
            self.assertEqual(trace["response_error"], error_body)
            self.assertEqual(trace["status_code"], 400)
            self.assertEqual(len(trace["request"]["assets"]), 6)
            self.assertTrue((folder / "06-failed-c0000.json").exists())
            self.assertFalse((folder / "06-check-judgments.json").exists())

    def test_body_limit_is_checked_locally_and_all_input_frames_remain_in_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            inputs, execution, packages = self.fixture(folder)
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch.object(ev, "MAX_BODY_BYTES", 100), \
                    patch.object(ev, "urlopen") as http:
                with self.assertRaisesRegex(ValueError, "body limit"):
                    self.judge(folder, inputs, execution, packages)
            http.assert_not_called()
            trace = json.loads((folder / "06-judge-c0000.json").read_text())
            self.assertGreater(trace["request_bytes"], 100)
            self.assertEqual([f["frame_id"] for f in trace["request"]["state"]["evidence"]["frames"]], [0, 1, 2])
            self.assertEqual(len(trace["request"]["assets"]), 6)
            self.assertFalse((folder / "06-check-judgments.json").exists())

    def test_judgment_does_not_rerun_experts_or_use_raw_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            inputs, execution, packages = self.fixture(folder)
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch.object(ev, "post", return_value=jev_response()) as post, \
                    patch.object(ev, "ObjectSpatialExpert") as detector, patch("expert_execution.ShortWindowMotionExpert") as motion, \
                    patch("expert_execution.pairwise_spatial_evidence") as geometry:
                self.judge(folder, inputs, execution, packages)
            for expert in (detector, motion, geometry): expert.assert_not_called()
            text = json.dumps(post.call_args.args[2]["state"])
            self.assertNotIn('"samples"', text)
            self.assertNotIn('"counts"', text)
            self.assertNotIn('"supporting_evidence_ids"', text)

    def test_zero_displacement_is_usable_and_prompt_keeps_temporal_and_physical_limits(self):
        packet = {"measurements": [{"expert_id": "short_window_motion", "summary": {
            "target": {"last_available_from_seed": {"magnitude_px": 0}}, "reference": None, "relative_motion": None}}]}
        self.assertTrue(cj.has_usable_measurements(packet))
        packet["measurements"][0]["summary"]["target"]["last_available_from_seed"] = None
        self.assertFalse(cj.has_usable_measurements(packet))
        for phrase in ("at least once", "episode end", "throughout", "Early achievement can later be undone",
                       "Images can be sufficient", "physical stability", "Do not grade"):
            self.assertIn(phrase, cj.JUDGMENT_PROMPT)

    def test_main_releases_capture_and_keeps_packages_on_judgment_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, execution, packages = self.fixture(root / "fixture-data")
            obs, selection, times, candidates, matching = inputs
            model, video = root / "model.onnx", root / "video.mp4"
            model.write_bytes(b"mocked")
            video.write_bytes(b"mocked")
            args = SimpleNamespace(video=video, yolo_model=model, out=root / "run", task="Task", split="test", episode="synthetic")
            cap = Mock()
            cap.get.side_effect = [16, 12]
            def package(*arguments):
                folder = arguments[4]
                ev.save(folder / "05-check-evidence.json", packages)
                self.write_images(folder, inputs)
                return packages
            with patch.object(ev, "frame_times", return_value=times), patch.object(ev.cv2, "VideoCapture", return_value=cap), \
                    patch.object(ev, "observe", return_value=obs), patch.object(ev, "select_windows", return_value=selection), \
                    patch.object(ev, "generate_candidates", return_value=candidates), patch.object(ev, "match_windows", return_value=matching), \
                    patch.object(ev, "execute_experts", return_value=execution), patch.object(ev, "package_evidence", side_effect=package), \
                    patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch.object(ev, "urlopen", side_effect=OSError("offline test error")), \
                    patch.object(ev, "score_saved") as score:
                with self.assertRaisesRegex(OSError, "offline test error"): ev.run(args)
            cap.release.assert_called_once()
            score.assert_not_called()
            self.assertTrue((args.out / "05-check-evidence.json").exists())
            self.assertTrue((args.out / "06-failed-c0000.json").exists())
            self.assertFalse((args.out / "06-check-judgments.json").exists())
            self.assertFalse((args.out / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
