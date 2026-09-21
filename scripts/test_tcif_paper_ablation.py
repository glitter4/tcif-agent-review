import json
import tempfile
import unittest
from pathlib import Path
from tcif_paper_ablation import (configuration, TCIF_ABLATIONS, ETAS, CHECKPOINTS,
                                 validate_summary, score_details, eval_command, summarize, eta_dir)


class ProtocolTests(unittest.TestCase):
    def test_locked_setting_and_only_intended_differences(self):
        output = Path("outputs").resolve()
        base = configuration("full", output)
        self.assertEqual((base["seed"], base["dropout"], base["epochs"]), (40,.3,10))
        allowed = {"tcif_ablation", "run_name", "factor_code", "save_dir", "log_dir"}
        for variant in TCIF_ABLATIONS:
            cfg = configuration(variant, output)
            differences = {k for k in base if cfg[k] != base[k]}
            expected = allowed | ({"tcif_enable_transition_gate", "tcif_transition_gate_loss_weight"}
                                  if variant in ("no_gate", "standard_context") else set())
            self.assertLessEqual(differences, expected)
            self.assertEqual(cfg["tcif_context_aux_weight"], .1)
            for eta in ETAS:
                cmd = eval_command("python", cfg, output, eta)
                self.assertEqual(float(cmd[cmd.index("--final_pred_eta")+1]),eta)
        self.assertEqual(len(TCIF_ABLATIONS)*len(ETAS)*len(CHECKPOINTS),70)

    def test_nonzero_macro_f1_rounding_and_mae(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"details.csv"
            path.write_text("true_value,pred_value\n0,-1\n-1,-0.5\n1,0\n1,-0.1\n",encoding="utf-8")
            metrics=score_details(path)
            self.assertEqual(metrics["samples"],4)
            self.assertEqual(metrics["nonzero_samples"],3)
            self.assertAlmostEqual(metrics["acc2non0"],200/3)
            self.assertAlmostEqual(metrics["f1non0"],200/3)
            self.assertEqual(metrics["acc7"],25.)
            self.assertAlmostEqual(metrics["mae"],.9)

    def test_complete_grid_and_failed_results_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            with self.assertRaises(FileNotFoundError):
                summarize(output)
            for variant in TCIF_ABLATIONS:
                for eta in ETAS:
                    destination=eta_dir(output,variant,eta)
                    destination.mkdir(parents=True)
                    rows=[]
                    for checkpoint in CHECKPOINTS:
                        (destination/f"{checkpoint}_test_details.csv").write_text(
                            "true_value,pred_value\n-1,-1\n1,1\n",encoding="utf-8")
                        rows.append(dict(model_stem=checkpoint,status="ok",output_dir=str(destination),
                            hparams=dict(tcif_ablation=variant,final_pred_eta=eta)))
                    path=destination/"all_models_summary.json"
                    path.write_text(json.dumps(rows),encoding="utf-8")
            report=summarize(output)
            self.assertEqual(len(report),10)
            self.assertTrue(all(len(row["sweep"])==7 for row in report))
            self.assertTrue(all(row["selected_eta"]["eta"]==.8 for row in report))
            with self.assertRaises(RuntimeError):
                validate_summary(path,variant,.8)  # last completed directory is eta=1
            rows[0]["status"]="error"
            path.write_text(json.dumps(rows),encoding="utf-8")
            with self.assertRaises(RuntimeError):
                summarize(output)


if __name__ == "__main__":
    unittest.main()
