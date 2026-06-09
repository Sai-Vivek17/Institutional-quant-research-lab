import unittest

import pandas as pd

from quant_research_lab.metrics import evaluate_performance, max_drawdown


class MetricsTest(unittest.TestCase):
    def test_max_drawdown_uses_compounded_equity(self) -> None:
        returns = pd.Series([0.10, -0.20, 0.05])
        self.assertAlmostEqual(max_drawdown(returns), -0.20)

    def test_evaluate_performance_outputs_headline_metrics(self) -> None:
        frame = pd.DataFrame(
            {
                "net_return": [0.01, -0.005, 0.002, 0.004],
                "turnover": [1.0, 0.4, 0.5, 0.2],
                "gross_exposure": [1.0, 1.0, 1.0, 1.0],
                "net_exposure": [0.0, 0.01, -0.01, 0.0],
                "transaction_cost": [0.0002, 0.0001, 0.0001, 0.00005],
            }
        )
        metrics = evaluate_performance(frame)
        self.assertEqual(metrics["days"], 4)
        self.assertIn("sharpe", metrics)
        self.assertGreater(metrics["hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
