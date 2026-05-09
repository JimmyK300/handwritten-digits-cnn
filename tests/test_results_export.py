import json
import tempfile
import unittest
from pathlib import Path

from myAPI.results_export import export_results


class _DummyDataset:
    def __len__(self):
        return 10

    def sample_id(self, idx: int) -> str:
        return f"/dummy/path/img_{int(idx)}.png"


class ExportResultsTests(unittest.TestCase):
    def test_export_writes_files_and_wrong_items(self):
        ds = _DummyDataset()

        wrong_details = [
            {"index": 1, "target": 7, "pred": 2},
            {"index": 3, "target": 0, "pred": 9},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            run_dir = export_results(
                out_dir=out_dir,
                run_name="test_run",
                train_loss_history=[0.5, 0.25],
                train_accuracy_history=[0.8, 0.9],
                test_loss=0.123,
                test_accuracy=0.456,
                confusion_matrix=[[1, 2], [3, 4]],
                wrong_details=wrong_details,
                train_dataset=ds,
                test_dataset=ds,
                extra={"note": "unit-test"},
            )

            self.assertTrue((run_dir / "results.json").exists())
            self.assertTrue((run_dir / "confusion_matrix.csv").exists())
            self.assertTrue((run_dir / "wrong_samples.txt").exists())

            results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
            self.assertIn("wrong", results)
            self.assertIn("items", results["wrong"])

            items = results["wrong"]["items"]
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0]["index"], 1)
            self.assertEqual(items[0]["target"], 7)
            self.assertEqual(items[0]["pred"], 2)
            self.assertEqual(items[0]["sample"], "/dummy/path/img_1.png")

            wrong_samples_txt = (
                (run_dir / "wrong_samples.txt").read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(
                wrong_samples_txt, ["/dummy/path/img_1.png", "/dummy/path/img_3.png"]
            )


if __name__ == "__main__":
    unittest.main()
