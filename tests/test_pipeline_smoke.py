import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from quant_research_lab.pipeline import run_pipeline


class PipelineSmokeTest(unittest.TestCase):
    def test_small_pipeline_runs_end_to_end(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            result = run_pipeline(
                config_overrides={
                    "data": {
                        "periods": 220,
                        "assets": 32,
                        "seed": 123,
                    },
                    "model": {
                        "train_window": 80,
                        "purge_days": 2,
                        "retrain_every": 10,
                        "min_train_observations": 1200,
                    },
                    "report": {
                        "enabled": False,
                        "output_dir": str(Path(temporary_directory) / "reports"),
                    },
                    "output": {
                        "save_intermediate": False,
                        "data_dir": str(Path(temporary_directory) / "data"),
                    },
                }
            )

        self.assertFalse(result.predictions.empty)
        self.assertFalse(result.positions.empty)
        self.assertFalse(result.backtest.empty)
        self.assertIn("sharpe", result.metrics)


if __name__ == "__main__":
    unittest.main()
