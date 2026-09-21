"""CPU checks: no datasets, downloaded backbones, or accelerator required."""
import argparse
import ast
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from models.tcif import build_task_filters, TemporalContextInnovationFilter
from tcif_ablation_config import TCIF_ABLATIONS, load_training_defaults, validate_training_ablation

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def pair(variant, feature=8, latent=4):
    return build_task_filters(variant, feature_dim=feature, latent_dim=latent,
        enable_transition_gate=variant not in ("no_gate", "standard_context"),
        transition_gate_hidden_dim=6 if feature == 8 else 64)


class AblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(40)
        self.local = torch.randn(3, 8)
        self.context = torch.randn(3, 2, 8)
        self.valid = torch.tensor([[1, 1], [1, 0], [0, 0]], dtype=torch.bool)
        self.pos = torch.tensor([[-1., 1.], [-1., 0.], [0., 0.]])

    def forward(self, layer):
        return layer(self.local, self.context, self.valid, self.pos)

    def test_zero_initialization_and_empty_context_for_every_variant(self):
        for name in TCIF_ABLATIONS:
            with self.subTest(name=name):
                reg, _ = pair(name)
                out = self.forward(reg)
                self.assertTrue(torch.equal(out["feature"], self.local))
                projection = reg.fusion[-1] if name == "standard_context" else reg.output_projection
                torch.nn.init.normal_(projection.weight)
                torch.nn.init.ones_(projection.bias)
                out = self.forward(reg)
                self.assertTrue(torch.isfinite(out["feature"]).all())
                self.assertTrue(torch.equal(out["feature"][2], self.local[2]))
                self.assertTrue(torch.equal(out["context_attention"][2], torch.zeros(2)))

    def test_equal_weight_changes_only_fusion(self):
        full, _ = pair("full")
        equal, _ = pair("equal_weight")
        equal.load_state_dict(full.state_dict(), strict=True)
        a, b = self.forward(full), self.forward(equal)
        for key in ("prior_mean", "prior_variance", "context_attention", "continuation_gate"):
            self.assertTrue(torch.equal(a[key], b[key]), key)
        self.assertTrue(torch.equal(b["posterior_mean"], .5*(b["local_mean"]+b["prior_mean"])))
        self.assertFalse(torch.allclose(a["posterior_mean"], b["posterior_mean"]))

    def test_no_gate_and_genuine_weight_sharing(self):
        reg, cls = pair("no_gate")
        self.assertIsNone(reg.transition_gate)
        self.assertTrue(torch.equal(self.forward(reg)["continuation_gate"], torch.tensor([1., 1., 0.])))
        reg, cls = pair("shared_filter")
        self.assertIs(reg, cls)
        self.assertEqual(len(list(torch.nn.ModuleList([reg, cls]).parameters())), len(list(reg.parameters())))

    def test_all_variants_preserve_subsequent_initialization(self):
        states = []
        for name in TCIF_ABLATIONS:
            torch.manual_seed(40)
            pair(name)
            states.append(torch.randn(16))
        self.assertTrue(all(torch.equal(states[0], x) for x in states[1:]))

    def test_save_restore_and_gradients(self):
        for name in TCIF_ABLATIONS:
            with self.subTest(name=name):
                reg, cls = pair(name)
                modules = torch.nn.ModuleDict({"reg": reg, "cls": cls})
                out = self.forward(reg)
                (out["feature"].square().mean() + .1*out["prior_mean"].square().mean()).backward()
                proj = reg.fusion[-1] if name == "standard_context" else reg.output_projection
                self.assertGreater(proj.weight.grad.abs().sum().item(), 0)
                stream = io.BytesIO()
                torch.save(modules.state_dict(), stream)
                stream.seek(0)
                a,b = pair(name)
                restored = torch.nn.ModuleDict({"reg": a, "cls": b})
                restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
                self.assertTrue(torch.equal(self.forward(a)["feature"], out["feature"]))
                self.assertEqual(a is b, name == "shared_filter")

    def test_standard_fusion_masking_and_parameter_match(self):
        reg,_ = pair("standard_context")
        a = self.forward(reg)
        changed = self.context.clone()
        changed[~self.valid] = 1e4
        b = reg(self.local, changed, self.valid, self.pos)
        self.assertTrue(torch.equal(a["prior_mean"], b["prior_mean"]))
        self.assertTrue(torch.isnan(a["prior_variance"]).all())
        full,_ = pair("full", 3072, 128)
        ordinary,_ = pair("standard_context", 3072, 128)
        n = sum(p.numel() for p in full.parameters())
        m = sum(p.numel() for p in ordinary.parameters())
        self.assertLess(abs(m-n)/n, .005)
        print(f"Active parameters per branch: full={n}, standard_context={m}")

    def test_archived_full_filter_compatibility(self):
        path = os.environ.get("TCIF_REFERENCE_MODULE")
        if not path:
            self.skipTest("Set TCIF_REFERENCE_MODULE to the untouched source tcif.py")
        spec = importlib.util.spec_from_file_location("reference_tcif", path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        torch.manual_seed(40)
        old = reference.TemporalContextInnovationFilter(8, 4, enable_transition_gate=True, transition_gate_hidden_dim=6)
        torch.manual_seed(40)
        new,_ = pair("full")
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
        torch.nn.init.normal_(old.output_projection.weight)
        new.load_state_dict(old.state_dict(), strict=True)
        a,b = self.forward(old), self.forward(new)
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_configuration_parser_and_gate_loss_validation(self):
        # Exercise the actual argparse declarations without importing backbones.
        source = ast.parse((ROOT / "server-code/train_emotion.py").read_text(encoding="utf-8"))
        block = next(n for n in source.body if isinstance(n, ast.If) and
                     isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__")
        calls = [n for n in block.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "add_argument"]
        parser = argparse.ArgumentParser()
        env = dict(parser=parser, argparse=argparse, os=os, STRUCTV7_REMOTE_ROOT="unused",
                   TEMPORAL_KERNEL_CHOICES=("legacy_exp",), TCIF_ABLATIONS=TCIF_ABLATIONS,
                   REG_CLS_MAG_CONSISTENCY_MODE="shrink_reg_to_detached_cls_same_sign_same_acc7_bin")
        for node in source.body:
            if isinstance(node, ast.Assign):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = value
        exec(compile(ast.Module(body=calls, type_ignores=[]), "<parser declarations>", "exec"), env)
        load_training_defaults(parser, ROOT / "configs/tcif_paper_single_model.json")
        args = parser.parse_args([])
        validate_training_ablation(args)
        self.assertEqual((args.seed,args.tcif_latent_dim,args.final_pred_eta,args.dropout), (40,128,.4,.3))
        args.tcif_ablation = "no_gate"
        with self.assertRaises(ValueError):
            validate_training_ablation(args)
        args.tcif_enable_transition_gate = False
        args.tcif_transition_gate_loss_weight = 0.
        validate_training_ablation(args)


class ModelIntegrationTests(unittest.TestCase):
    def test_each_variant_forward_backward_and_checkpoint_metadata(self):
        from test_signed_reg_cls7_head import build_tiny_model, tiny_batch
        from test_tcif import _tiny_tcif_context
        from train_emotion import _compute_tcif_context_aux_loss, _write_checkpoint_hparams
        reference_state = None
        for name in TCIF_ABLATIONS:
            with self.subTest(name=name):
                torch.manual_seed(40)
                model = build_tiny_model("signed_reg_cls7", enable_tcif=True, tcif_ablation=name,
                    tcif_latent_dim=4, tcif_enable_transition_gate=name not in ("no_gate","standard_context"))
                shared_state = {k:v for k,v in model.state_dict().items() if not k.startswith(("tcif_regression.","tcif_ordinal."))}
                if reference_state is None:
                    reference_state = shared_state
                for k,v in shared_state.items():
                    self.assertTrue(torch.equal(v, reference_state[k]), k)
                batch = tiny_batch()
                out = model(batch["image"], batch["input_ids"], batch["attention_mask"],
                            batch["audio_values"], batch["audio_attention_mask"], **_tiny_tcif_context(batch))
                args = SimpleNamespace(reg_loss_type="smooth_l1", tcif_context_aux_cls7_weight=.5,
                                       tcif_ablation=name, lr=6e-5, use_text_cache=False)
                aux,_,_,valid = _compute_tcif_context_aux_loss(out,batch["raw_valence"],args)
                self.assertEqual(valid,2)
                loss = out["y_reg"].square().mean()+out["cls7_logits"].square().mean()+.1*aux
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertEqual(out["extras"]["tcif"]["ablation"],name)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"checkpoint.json"
                    _write_checkpoint_hparams(path,args,checkpoint_epoch=1)
                    self.assertEqual(json.loads(path.read_text())["tcif_ablation"],name)


if __name__ == "__main__":
    unittest.main()
