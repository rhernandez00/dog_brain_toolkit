"""Exercise class jobs through construction, queue promotion, and worker commands."""

import ast
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import job_status
import schedule_rsa
from run_jobs import build_command
from scheduler.dag import CLASS_STEPS, build_job_graph, build_single_job
from scheduler.jobs import claim_job, complete_job, create_job, load_job
from scheduler.paths import get_queue_dir
from tools.schedule_steps import parse_steps


def class_jobs(rsa_class):
    return [
        job
        for target_step in (11.3, 11.5)
        for job in build_job_graph(
            dataset="EmoC", model="basic-block", rsa_model=None, specie="D",
            start_step=11.2, target_step=target_step, rsa_class=rsa_class,
            radius=4, reps=100, reps_group=1000,
        )
    ]


class SchedulerClassTests(unittest.TestCase):
    def test_notebook_cell_dry_run_plans_twelve_jobs(self):
        cell = Path(__file__).resolve().parent / "tools" / "schedule_leave_one_out_cell.py"
        tree = ast.parse(cell.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "DRY_RUN"
                for target in node.targets
            ):
                node.value = ast.Constant(True)
        namespace = {}
        with redirect_stdout(io.StringIO()):
            exec(compile(ast.fix_missing_locations(tree), str(cell), "exec"), namespace)
        self.assertEqual(namespace["created"], 0)
        self.assertIsNone(namespace["queue_dir"])
        jobs = list(namespace["jobs_by_id"].values())
        self.assertEqual(len(jobs), 12)
        self.assertEqual({job["rsa_class"] for job in jobs}, {"all", "dog", "hum"})
        self.assertEqual(sum(job["status"] == "waiting" for job in jobs), 6)

    def test_fractional_class_jobs_roundtrip_and_worker_arguments(self):
        jobs = [job for name in ("all", "dog", "hum") for job in class_jobs(name)]
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({job["job_id"] for job in jobs}), 12)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            queue = get_queue_dir(temp)
            for job in jobs:
                create_job(queue, job)
                saved = load_job(queue / job["status"] / (job["job_id"] + ".json"))
                self.assertEqual(saved["step"], job["step"])
                command = build_command(saved, "C:/github", "python", "C:/markers")
                for flag, value in (("--steps_to_run", str(job["step"])),
                                    ("--rsa_class", job["rsa_class"]),
                                    ("--radius", "4"), ("--reps", "100"),
                                    ("--reps_group", "1000")):
                    self.assertEqual(command[command.index(flag) + 1], value)
                self.assertNotIn("--rsa_model", command)
            self.assertEqual(len(list((queue / "pending").glob("*.json"))), 6)
            self.assertEqual(len(list((queue / "waiting").glob("*.json"))), 6)

    def test_only_matching_class_and_branch_promotes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            queue = get_queue_dir(temp)
            jobs = [job for name in ("all", "dog", "hum") for job in class_jobs(name)]
            for job in jobs:
                create_job(queue, job)
            path, claimed = claim_job(queue)
            self.assertEqual(claimed["step"], 11.2)
            complete_job(queue, path, claimed)
            for job in jobs:
                if job["step"] not in (11.3, 11.5):
                    continue
                state = ("pending" if job["step"] == 11.3
                         and job["rsa_class"] == claimed["rsa_class"] else "waiting")
                self.assertTrue((queue / state / (job["job_id"] + ".json")).exists())

    def test_single_jobs_require_class(self):
        for step in CLASS_STEPS:
            with self.subTest(step=step):
                with self.assertRaisesRegex(ValueError, "requires rsa_class"):
                    build_single_job("EmoC", "basic-block", None, "D", step)
                job = build_single_job("EmoC", "basic-block", None, "D", step,
                                       rsa_class="dog")
                self.assertIn(f"step{step}", job["job_id"])
                self.assertEqual(job["rsa_class"], "dog")
                job["rsa_class"] = None
                with self.assertRaisesRegex(ValueError, "requires rsa_class"):
                    build_command(job, "C:/github", "python", "C:/markers")

    def test_graph_can_include_pairwise_dependencies(self):
        jobs = build_job_graph("EmoC", "basic-block", None, "D", start_step=1,
                               target_step=11.5, rsa_class="all")
        self.assertEqual([job["step"] for job in jobs], [1, 11.4, 11.5])
        self.assertEqual(jobs[1]["deps"], [jobs[0]["job_id"]])
        self.assertEqual(jobs[2]["deps"], [jobs[1]["job_id"]])
        self.assertEqual(parse_steps("11.2,11.3,11.4,11.5"), sorted(CLASS_STEPS))

    def test_cli_schedules_and_reports_fractional_class_jobs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            with patch("sys.argv", ["schedule_rsa.py", "--dataset", "EmoC",
                                    "--rsa_class", "dog", "--specie", "D",
                                    "--start_step", "11.2", "--target_step", "11.5",
                                    "--radius", "4"]), \
                    patch.object(schedule_rsa, "get_paths", return_value=(temp, "", "")), \
                    patch("builtins.print"):
                schedule_rsa.main()
            queue = Path(temp) / "job_queue"
            self.assertEqual(len(list((queue / "pending").glob("*.json"))), 1)
            self.assertEqual(len(list((queue / "waiting").glob("*.json"))), 1)
            with patch("sys.argv", ["job_status.py", "--dataset", "EmoC",
                                    "--rsa_class", "dog", "--radius", "4"]), \
                    patch.object(job_status, "get_paths", return_value=(temp, "", "")), \
                    patch("builtins.print") as output:
                job_status.main()
            self.assertTrue(any("READY" in str(call) for call in output.call_args_list))


if __name__ == "__main__":
    unittest.main()
