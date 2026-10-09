"""Radius-specific scheduler IDs and worker arguments."""

import tempfile
import unittest
from pathlib import Path

from run_jobs import build_command
from scheduler.dag import build_job_graph, build_single_job
from scheduler.jobs import create_job, load_job


class SchedulerRadiusTests(unittest.TestCase):
    def make_job(self, specie="D", radius=None):
        return build_single_job(
            dataset="EmoC", model="basic-block", rsa_model="valence3__all",
            specie=specie, step=5, radius=radius,
        )

    def test_species_defaults_keep_historical_ids(self):
        for specie, default in (("D", 3), ("H", 4)):
            with self.subTest(specie=specie):
                legacy = self.make_job(specie)["job_id"]
                self.assertEqual(self.make_job(specie, default)["job_id"], legacy)
                self.assertNotIn("__rad", legacy)

    def test_non_default_radius_has_distinct_queue_id_and_worker_argument(self):
        default = self.make_job()
        radius_four = self.make_job(radius=4)
        self.assertNotEqual(default["job_id"], radius_four["job_id"])
        self.assertIn("__rad4", radius_four["job_id"])
        with tempfile.TemporaryDirectory() as temp:
            queue = Path(temp)
            (queue / "pending").mkdir()
            create_job(queue, default)
            create_job(queue, radius_four)
            files = list((queue / "pending").glob("*.json"))
            self.assertEqual(len(files), 2)
            self.assertEqual({load_job(f)["radius"] for f in files}, {None, 4})
        command = build_command(radius_four, "C:/repo", "python", "C:/markers")
        self.assertEqual(command[command.index("--radius") + 1], "4")

    def test_graph_dependencies_share_radius(self):
        jobs = build_job_graph(
            "EmoC", "basic-block", "valence3__all", "D",
            start_step=4, target_step=6, radius=4,
        )
        ids = {job["job_id"] for job in jobs}
        self.assertTrue(all("__rad4" in job_id for job_id in ids))
        self.assertTrue(all(job["radius"] == 4 for job in jobs))
        self.assertTrue(all(dep in ids for job in jobs for dep in job["deps"]))


if __name__ == "__main__":
    unittest.main()
