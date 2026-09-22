"""Regression checks for condition-wise human and dog FEAT designs."""
import re
import tempfile
import unittest
from pathlib import Path

from utils import generate_fsf


ROOT = Path(__file__).resolve().parents[1]


class GenerateFsfTests(unittest.TestCase):
    def test_species_templates(self):
        for template_name, convolution in (
            ("basic_H.fsf", "3"),
            ("basic_DHRF_modified.fsf", "2"),
        ):
            for n in (1, 3, 12):
                for explicit_files in (False, True):
                    with self.subTest(template=template_name, n=n,
                                      explicit_files=explicit_files):
                        template = ROOT / "FSL_designs" / template_name
                        conditions = [f"condition_{i}" for i in range(1, n + 1)]
                        ev_files = [f"/onsets/{name}.txt" for name in conditions]
                        with tempfile.TemporaryDirectory() as directory:
                            output = Path(directory) / "design.fsf"
                            generate_fsf(n, template, output, conditions,
                                         ev_files if explicit_files else None)
                            text = output.read_text(encoding="utf-8")
                        assignments = re.findall(
                            r"^set (\S+) (.*)$", text, re.MULTILINE)
                        settings = dict(assignments)
                        self.assertEqual(len(assignments), len(settings),
                                         "Duplicate FSF settings")
                        self.assertNotIn("NUM", text)
                        for key, expected in (("evs_orig", n), ("evs_real", 2 * n),
                                              ("ncon_orig", n), ("ncon_real", n)):
                            self.assertEqual(settings[f"fmri({key})"], str(expected))
                        self.assertEqual(sum(key.startswith("fmri(evtitle")
                                             for key in settings), n)
                        for i, name in enumerate(conditions, 1):
                            self.assertEqual(settings[f"fmri(evtitle{i})"], f'"{name}"')
                            self.assertEqual(settings[f"fmri(convolve{i})"], convolution)
                            self.assertEqual(settings[f"fmri(deriv_yn{i})"], "1")
                            if convolution == "2":
                                self.assertEqual(settings[f"fmri(gammasigma{i})"], "1.5")
                                self.assertEqual(settings[f"fmri(gammadelay{i})"], "3")
                            custom = settings[f"fmri(custom{i})"]
                            self.assertTrue(custom.endswith(f'/{name}.txt"'))
                            if explicit_files:
                                self.assertEqual(custom, f'"{ev_files[i - 1]}"')
                            for mode, width, active in (("orig", n, i),
                                                        ("real", 2 * n, 2 * i - 1)):
                                weights = [float(settings[f"fmri(con_{mode}{i}.{j})"])
                                           for j in range(1, width + 1)]
                                self.assertEqual(weights, [float(j == active)
                                                          for j in range(1, width + 1)])
                        # Inputs and non-GUI settings must survive unchanged.
                        original = dict(re.findall(r"^set (\S+) (.*)$",
                                                   template.read_text(), re.MULTILINE))
                        for key in ("feat_files(1)", "confoundev_files(1)",
                                    "fmri(regstandard)", "fmri(overwrite_yn)"):
                            self.assertEqual(settings[key], original[key])


if __name__ == "__main__":
    unittest.main()
