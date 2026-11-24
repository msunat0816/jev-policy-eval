"""Binding completion and end-to-end evidence retention; no paid model calls."""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from test_vlm_matching import candidate_bundle, candidate_frames, matching_observation, window_matching
from test_vlm_matching_integration import ev, FrameCapture, qwen_response
from vlm_matching import parse_window_matching, required_binding_pairs
from window_selection import summarize_selection


class BindingGapTests(unittest.TestCase):
    def test_program_slots_are_shared_and_motion_needs_only_seed_bindings(self):
        value = window_matching()
        value["checks"][0]["experts"][1].update(motion_mode="relative", reference_entity_key="bowl")
        value["checks"][1]["experts"] = copy.deepcopy(value["checks"][0]["experts"])
        self.assertEqual(required_binding_pairs(value["checks"]),
                         [("apple", 0), ("apple", 2), ("bowl", 0), ("bowl", 2)])

    def test_gaps_defer_only_omissions_and_saved_inputs_stay_strict(self):
        value = window_matching()
        value["bindings"].pop(0)
        gaps = []
        parsed = parse_window_matching(json.dumps(value), {"id": "w0"}, matching_observation()["candidate_checks"],
                                       [0, 1, 2], candidate_frames([0, 1, 2]), binding_gaps=gaps)
        self.assertEqual(gaps, [("apple", 0)])
        self.assertEqual(parsed, value)
        with self.assertRaises(ValueError):
            parse_window_matching(json.dumps(parsed), {"id": "w0"}, matching_observation()["candidate_checks"],
                                  [0, 1, 2], candidate_frames([0, 1, 2]))
        for mode in ("duplicate", "invalid_candidate", "motion_gap"):
            with self.subTest(mode=mode):
                bad = copy.deepcopy(value)
                if mode == "duplicate": bad["bindings"].append(copy.deepcopy(bad["bindings"][0]))
                elif mode == "invalid_candidate": bad["bindings"][0]["candidate_id"] = "f0:d999"
                else: bad["checks"][0]["experts"][1]["frame_ids"] = [0, 2]
                gaps = []
                with self.assertRaises(ValueError):
                    parse_window_matching(json.dumps(bad), {"id": "w0"}, matching_observation()["candidate_checks"],
                                          [0, 1, 2], candidate_frames([0, 1, 2]), binding_gaps=gaps)
                self.assertEqual(gaps, [])


@unittest.skipIf(ev is None, "OpenCV and NumPy required")
class BindingRecoveryTests(unittest.TestCase):
    def inputs(self, *, two_windows=False):
        obs = matching_observation()
        selection = summarize_selection([(0, 2), (2, 4)], {"w0": "p2", "w1": "p2" if two_windows else "p1"})
        times = [i / 10 for i in range(5)]
        return obs, selection, times, candidate_bundle(selection, times)

    def match(self, folder, initial, completion, *, two_windows=False):
        inputs = self.inputs(two_windows=two_windows)
        replies = [qwen_response(initial), completion]
        if two_windows:
            replies.append(qwen_response(window_matching("w1", 2, 4, motion=False)))
        def transport(request, timeout):
            reply = replies.pop(0)
            if isinstance(reply, Exception): raise reply
            return io.BytesIO(json.dumps(reply).encode())
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only"}), \
                patch.object(ev, "urlopen", side_effect=transport) as http:
            matching = ev.match_windows("Task", *inputs[:3], FrameCapture(), folder, inputs[3])
        return (*inputs, matching), http

    def test_completion_uses_only_program_slots_and_preserves_raw_traces_and_good_bindings(self):
        initial = window_matching(motion=False)
        supplied = initial["bindings"].pop(0)
        reply = {"window_id": "w0", "bindings": [supplied]}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            inputs, http = self.match(folder, initial, qwen_response(reply))
            self.assertEqual(http.call_count, 2)
            window = inputs[-1]["windows"][0]
            self.assertEqual(window["checks"], initial["checks"])
            self.assertEqual(window["binding_completion"]["status"], "completed")
            self.assertEqual(window["binding_completion"]["requested_pairs"], [["apple", 0]])
            repair = json.loads((folder / "03-bind-w0.json").read_text())
            parts = repair["request"]["messages"][0]["content"]
            context = json.loads(parts[0]["text"].split("\nContext: ", 1)[1])
            self.assertEqual(context["requested_pairs"], [["apple", 0]])
            self.assertEqual(context["existing_bindings"], initial["bindings"])
            self.assertEqual([f["frame_id"] for f in context["candidate_frames"]], [0])
            self.assertEqual(sum(p["type"] == "image_url" for p in parts), 2)
            raw = json.loads((folder / "03-match-w0.json").read_text())
            self.assertEqual(json.loads(raw["response"]["choices"][0]["message"]["content"]), initial)
            repaired = next(b for b in window["bindings"] if b["entity_key"] == "apple" and b["frame_id"] == 0)
            self.assertEqual(repaired["source"]["method"], "matching_vlm_binding_completion")
            self.assertEqual(repaired["source"]["request_file"], "03-bind-w0.json")
            from expert_execution import execute_proposals
            result = execute_proposals("Task", *inputs[:3], FrameCapture(), *inputs[3:])
            self.assertEqual(result["records"][0]["status"], "complete")

    def test_explicit_model_missing_is_distinct_from_program_unresolved(self):
        initial = window_matching(motion=False)
        binding = initial["bindings"].pop(0)
        binding.update(status="missing", candidate_id=None, reason="The candidate identity is ambiguous.")
        with tempfile.TemporaryDirectory() as tmp:
            inputs, _ = self.match(Path(tmp), initial, qwen_response({"window_id": "w0", "bindings": [binding]}))
            self.assertEqual(inputs[-1]["windows"][0]["binding_completion"]["status"], "completed")
            from expert_execution import execute_proposals
            result = execute_proposals("Task", *inputs[:3], FrameCapture(), *inputs[3:])
            self.assertEqual(result["records"][0]["execution_status"], "executed")
            self.assertEqual(result["records"][0]["status"], "partial")

    def test_partial_completion_skips_only_affected_expert_and_preserves_other_windows(self):
        initial = window_matching()
        first = initial["bindings"].pop(0)
        initial["bindings"].pop()  # Omit bowl at the spatial endpoint as well.
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            inputs, http = self.match(folder, initial, qwen_response({"window_id": "w0", "bindings": [first]}), two_windows=True)
            self.assertEqual(http.call_count, 3)
            self.assertEqual(inputs[-1]["windows"][0]["binding_completion"]["status"], "partial")
            from expert_execution import execute_proposals
            with patch("expert_execution._motion", return_value={"execution_status": "executed", "status": "complete", "reason": None, "result": {}}) as motion:
                result = execute_proposals("Task", *inputs[:3], FrameCapture(), *inputs[3:])
            self.assertEqual(motion.call_count, 1)
            self.assertEqual(result["records"][0]["reason"], "binding_unresolved")
            self.assertEqual(result["records"][2]["status"], "complete")
            self.assertEqual(inputs[-1]["windows"][0]["checks"], initial["checks"])
            self.assertEqual(len(inputs[-1]["windows"][0]["images"]), 3)

    def test_failed_completion_retains_images_through_jev_and_final_grading(self):
        from test_check_judgment import jev_response
        from test_final_evaluation import final_response
        initial = window_matching()
        initial["bindings"] = []
        for response in (URLError("write timed out"), qwen_response({"window_id": "w0", "bindings": []}),
                         qwen_response([])):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                inputs, http = self.match(folder, initial, response)
                self.assertEqual(http.call_count, 2)
                from expert_execution import execute_proposals
                with patch("expert_execution.pairwise_spatial_evidence") as spatial, patch("expert_execution.ShortWindowMotionExpert") as motion:
                    execution = execute_proposals("Task", *inputs[:3], FrameCapture(), *inputs[3:])
                spatial.assert_not_called()
                motion.assert_not_called()
                self.assertTrue(all(r["execution_status"] == "not_executed" for r in execution["records"]))
                packages = ev.build_evidence_packages("Task", *inputs[:3], *inputs[3:], execution)
                packet = packages["packages"][0]
                self.assertEqual([image["frame_id"] for image in packet["images"]], [0, 1, 2])
                self.assertTrue(all(m["summary"] is None and m["reason"] == "binding_unresolved" for m in packet["measurements"]))
                with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch.object(ev, "post", return_value=jev_response()) as jev:
                    judgments = ev.judge_checks("Task", *inputs[:3], folder, *inputs[3:], execution, packages)
                self.assertEqual(judgments["judgments"][0]["answer"], "yes")
                self.assertEqual(len(jev.call_args.args[2]["assets"]), 6)
                for choice in ("level_4", "insufficient_evidence"):
                    with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch.object(ev, "post", return_value=final_response(choice)):
                        final = ev.evaluate_final("Task", *inputs[:3], folder, *inputs[3:], execution, packages, judgments)
                    self.assertEqual(final["jev_choice"], choice)
                materials = json.loads((folder / "07-final-materials.json").read_text())
                self.assertIn("unresolved_binding", [r["kind"] for r in materials["checks"][0]["unresolved"]])

    def test_invalid_completion_is_rejected_atomically_and_cannot_replace_valid_bindings(self):
        initial = window_matching(motion=False)
        omitted = initial["bindings"].pop(0)
        for mode in ("other_slot", "invalid_candidate", "duplicate", "wrong_window", "proposals", "unresolved"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                reply = {"window_id": "w0", "bindings": [copy.deepcopy(omitted)]}
                if mode == "other_slot": reply["bindings"].append(copy.deepcopy(initial["bindings"][0]))
                elif mode == "invalid_candidate": reply["bindings"][0]["candidate_id"] = "f0:d999"
                elif mode == "duplicate": reply["bindings"].append(copy.deepcopy(omitted))
                elif mode == "wrong_window": reply["window_id"] = "w1"
                elif mode == "proposals": reply["checks"] = []
                else: reply["bindings"][0].update(status="unresolved", candidate_id=None)
                inputs, _ = self.match(Path(tmp), initial, qwen_response(reply))
                window = inputs[-1]["windows"][0]
                self.assertEqual(window["binding_completion"]["error"]["kind"], "invalid_response")
                self.assertEqual(window["bindings"][-1]["status"], "unresolved")
                self.assertEqual([{key: b[key] for key in initial["bindings"][0]} for b in window["bindings"][:-1]], initial["bindings"])

    def test_saved_recovery_cannot_forge_sources_slots_or_measurements(self):
        initial = window_matching()
        initial["bindings"] = []
        with tempfile.TemporaryDirectory() as tmp:
            inputs, _ = self.match(Path(tmp), initial, URLError("read timed out"))
            from expert_execution import execute_proposals, validate_execution_inputs
            execution = execute_proposals("Task", *inputs[:3], FrameCapture(), *inputs[3:])
            for mode in ("source", "model", "no_attempt", "slots", "duplicate", "non_null_candidate", "missing_claim"):
                with self.subTest(mode=mode):
                    changed = copy.deepcopy(inputs)
                    window = changed[-1]["windows"][0]
                    binding = window["bindings"][0]
                    if mode == "source": binding["source"]["request_file"] = "03-match-w0.json"
                    elif mode == "model": binding["source"]["model"] = ev.MATCHING_MODEL
                    elif mode == "no_attempt": window.pop("binding_completion")
                    elif mode == "slots": window["binding_completion"]["required_pairs"].pop()
                    elif mode == "duplicate": window["binding_completion"]["requested_pairs"].append(["apple", 0])
                    elif mode == "non_null_candidate": binding["candidate_id"] = "f0:d0"
                    else: binding["status"] = "missing"
                    with self.assertRaises(ValueError):
                        validate_execution_inputs("Task", *changed[:3], *changed[3:])
            execution["records"][0].update(execution_status="executed", result={})
            with self.assertRaisesRegex(ValueError, "unresolved binding must not have expert measurements"):
                ev.build_evidence_packages("Task", *inputs[:3], *inputs[3:], execution)

    def test_call_budget_and_initial_invalid_proposals_still_stop_without_recovery(self):
        initial = window_matching()
        initial["bindings"] = []
        for invalid_proposal in (False, True):
            with self.subTest(invalid_proposal=invalid_proposal), tempfile.TemporaryDirectory() as tmp:
                value = copy.deepcopy(initial)
                if invalid_proposal: value["checks"][0]["experts"][1]["frame_ids"] = [0, 2]
                inputs = self.inputs()
                with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only"}), \
                        patch.object(ev, "post", side_effect=[qwen_response(value), ValueError("configured per-episode API call limit reached")]) as post:
                    with self.assertRaises(ValueError):
                        ev.match_windows("Task", *inputs[:3], FrameCapture(), Path(tmp), inputs[3])
                self.assertEqual(post.call_count, 1 if invalid_proposal else 2)

    def test_worker_deadline_is_not_swallowed_as_a_completion_transport_failure(self):
        initial = window_matching()
        initial["bindings"] = []
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TimeoutError, "worker wall-time limit reached"):
                self.match(Path(tmp), initial, TimeoutError("worker wall-time limit reached"))
            self.assertFalse((Path(tmp) / "03-window-matching.json").exists())

    def test_empty_inventory_normalization_and_nonempty_completion_share_one_window(self):
        inputs = self.inputs()
        inputs[3]["frames"][2] = candidate_frames([2], inputs[2], empty=True)[0]
        initial = window_matching(motion=False)
        initial["bindings"] = []
        completion = {"window_id": "w0", "bindings": window_matching(motion=False)["bindings"][:2]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only"}), \
                patch.object(ev, "post", side_effect=[qwen_response(initial), qwen_response(completion)]) as post:
            matching = ev.match_windows("Task", *inputs[:3], FrameCapture(), Path(tmp), inputs[3])
        self.assertEqual(post.call_count, 2)
        window = matching["windows"][0]
        self.assertEqual(window["binding_completion"]["requested_pairs"], [["apple", 0], ["bowl", 0]])
        self.assertEqual({b["source"]["method"] for b in window["bindings"]},
                         {"program_empty_candidate_inventory", "matching_vlm_binding_completion"})
        from expert_execution import execute_proposals
        result = execute_proposals("Task", *inputs[:3], FrameCapture(), inputs[3], matching)
        self.assertEqual(result["records"][0]["status"], "partial")


if __name__ == "__main__":
    unittest.main()
