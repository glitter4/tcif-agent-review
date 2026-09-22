import os
import csv
import math
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.emotion_dataset import CMUMOSEIProcessDataset, CMUMOSIProcessDataset, _valence_to_7class, _valence_to_binary
from models.models_emotion import EmotionM4OE
from head_utils import cls7_expected_value, compute_final_prediction, continuous_to_cls7_hard, get_cls7_centers
from tqdm import tqdm

def metrics_from_lists(labels, preds, num_classes):
    if len(labels) == 0:
        return 0.0, 0.0, [0.0 for _ in range(num_classes)], [0.0 for _ in range(num_classes)]

    conf = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for y, p in zip(labels, preds):
        yi = int(y)
        pi = int(p)
        if 0 <= yi < num_classes and 0 <= pi < num_classes:
            conf[yi][pi] += 1

    per_class_precision = []
    per_class_recall = []
    f1s = []
    correct = 0
    total = 0

    for i in range(num_classes):
        tp = conf[i][i]
        fn = sum(conf[i][j] for j in range(num_classes)) - tp
        fp = sum(conf[j][i] for j in range(num_classes)) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_precision.append(prec)
        per_class_recall.append(rec)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

        correct += tp
        total += tp + fn

    acc = correct / total if total > 0 else 0.0
    macro_f1 = sum(f1s) / num_classes if num_classes > 0 else 0.0
    return acc, macro_f1, per_class_precision, per_class_recall

def _rankdata(values):
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0 for _ in range(n)]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks

def _pearsonr(xs, ys):
    n = len(xs)
    if n == 0:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = 0.0
    denom_x = 0.0
    denom_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        denom_x += dx * dx
        denom_y += dy * dy
    denom = math.sqrt(denom_x) * math.sqrt(denom_y)
    if denom == 0.0:
        return 0.0
    return num / denom

def regression_metrics(preds, targets):
    if len(preds) == 0:
        return 0.0, 0.0, 0.0, 0.0
    diffs = [p - t for p, t in zip(preds, targets)]
    mse = sum(d * d for d in diffs) / len(diffs)
    rmse = math.sqrt(mse)
    mae = sum(abs(d) for d in diffs) / len(diffs)
    pearson = _pearsonr(preds, targets)
    ranks_p = _rankdata(preds)
    ranks_t = _rankdata(targets)
    spearman = _pearsonr(ranks_p, ranks_t)
    return rmse, mae, pearson, spearman

def _embedding_cache_kwargs(batch, device):
    kwargs = {}
    for batch_key, model_key in (
        ("cached_vision", "cached_vision"),
        ("cached_text", "cached_text"),
        ("cached_audio", "cached_audio"),
        ("cached_text_mask", "cached_text_mask"),
        ("cached_audio_mask", "cached_audio_mask"),
    ):
        value = batch.get(batch_key)
        if value is not None:
            kwargs[model_key] = value.to(device, non_blocking=True)
    return kwargs

def _extract_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = state.get(key)
            if isinstance(value, dict):
                return value
    return state

def _state_has_adaptive_vit_context_gate(state_dict):
    if not isinstance(state_dict, dict):
        return False
    return any(
        str(key).endswith("vit_context_lambda_gate.weight")
        or str(key).endswith("vit_context_lambda_gate.bias")
        or str(key).endswith("vit_context_lambda_gate_norm.weight")
        for key in state_dict.keys()
    )

def _state_has_attention_vit_context_gate(state_dict):
    if not isinstance(state_dict, dict):
        return False
    return any(
        "vit_context_lambda_attn_q" in str(key)
        or "vit_context_lambda_attn_k" in str(key)
        for key in state_dict.keys()
    )

def _infer_vit_context_attention_dim(state_dict, default_dim: int):
    if not isinstance(state_dict, dict):
        return int(default_dim)
    for key, value in state_dict.items():
        if str(key).endswith("vit_context_lambda_attn_q.weight") and hasattr(value, "shape") and len(value.shape) >= 1:
            return int(value.shape[0])
    return int(default_dim)

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    default_tokenizer_path = "/path/to/user/models/google-bert_bert-base-uncased"
    if getattr(args, "bert_backbone", "vanilla") == "twitter_roberta":
        args.bert_model_path = args.bert_model_path_twitter_roberta
        if args.tokenizer_path == default_tokenizer_path:
            args.tokenizer_path = args.bert_model_path
    if getattr(args, "vit_backbone", "vanilla") == "fer":
        args.vit_model_path = args.vit_model_path_fer

    if args.dataset == "cmumosi":
        test_dataset = CMUMOSIProcessDataset(
            root=args.dataset_root,
            split="test",
            transform=transform,
            tokenizer_path=args.tokenizer_path,
            local_files_only=args.local_files_only,
            frame_policy=args.frame_policy,
            audio_sample_rate=args.audio_sample_rate,
            audio_max_seconds=args.audio_max_seconds,
            audio_frame_ms=args.audio_frame_ms,
            audio_hop_ms=args.audio_hop_ms,
            audio_denoise=not args.disable_audio_denoise,
            num_frames=args.num_frames,
            vit_context_ratio=args.vit_context_ratio,
        )
    else:
        test_dataset = CMUMOSEIProcessDataset(
            root=args.dataset_root,
            split="test",
            transform=transform,
            tokenizer_path=args.tokenizer_path,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            frame_policy=args.frame_policy,
            audio_sample_rate=args.audio_sample_rate,
            audio_max_seconds=args.audio_max_seconds,
            audio_frame_ms=args.audio_frame_ms,
            audio_hop_ms=args.audio_hop_ms,
            audio_denoise=not args.disable_audio_denoise,
            num_frames=args.num_frames,
            vit_context_ratio=args.vit_context_ratio,
            embedding_cache_root=args.embedding_cache_root,
        )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Test set size: {len(test_dataset)}")

    if os.path.exists(args.checkpoint):
        state_dict = _extract_state_dict(torch.load(args.checkpoint, map_location=device))
        if _state_has_attention_vit_context_gate(state_dict):
            if args.vit_context_lambda_mode != "attention":
                args.vit_context_lambda_mode = "attention"
                print("Detected attention ViT context lambda gate in checkpoint; using --vit_context_lambda_mode attention")
            args.vit_context_attention_dim = _infer_vit_context_attention_dim(state_dict, args.vit_context_attention_dim)
        elif args.vit_context_lambda_mode == "static" and _state_has_adaptive_vit_context_gate(state_dict):
            args.vit_context_lambda_mode = "adaptive"
            print("Detected adaptive ViT context lambda gate in checkpoint; using --vit_context_lambda_mode adaptive")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    num_polarity_classes = 2
    model = EmotionM4OE(
        num_classes=num_polarity_classes,
        num_aux_classes=num_polarity_classes,
        embed_dim=768,
        depth_msoe=2,
        depth_mtoe=1,
        num_experts_msoe=args.num_experts_msoe,
        num_experts_mtoe=args.num_experts_mtoe,
        num_shared_experts=args.num_shared_experts,
        num_text_specific_experts=args.num_text_specific_experts,
        num_audio_specific_experts=args.num_audio_specific_experts,
        num_vision_specific_experts=args.num_vision_specific_experts,
        enable_text_shared_experts=args.enable_text_shared_experts,
        vision_backbone_type=args.vision_backbone_type,
        vision_backbone_path=args.vision_backbone_path,
        vit_model_path=args.vit_model_path,
        bert_model_path=args.bert_model_path,
        hubert_model_path=args.hubert_model_path,
        local_files_only=args.local_files_only,
        dropout=args.dropout,
        enable_expert_load_output=args.enable_expert_load_output,
        vit_context_ratio=args.vit_context_ratio,
        vit_context_lambda_mode=args.vit_context_lambda_mode,
        vit_context_lambda_init=args.vit_context_lambda_init,
        vit_context_attention_dim=args.vit_context_attention_dim,
        output_head_mode=args.output_head_mode,
        final_pred_eta=args.final_pred_eta,
    ).to(device)

    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {args.checkpoint}")

    model.eval()
    model.reset_expert_load_stats()
    polarity_preds = []
    polarity_labels = []
    intensity_preds = []
    intensity_targets = []
    output_rows = []
    final_scores = []
    raw_valences = []
    reg_scores = []
    cls_expected_scores = []
    cls7_preds = []
    cls7_labels = []

    def _polarity_sign(idx):
        if int(idx) == 0:
            return -1
        return 1

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_values = batch["audio_values"].to(device)
            audio_attention_mask = batch["audio_attention_mask"].to(device)

            intensity = batch["intensity"].to(device)
            polarity = batch["polarity"].to(device)
            raw_valence = batch["raw_valence"].cpu().numpy().tolist()

            model_out = model(
                images,
                input_ids,
                attention_mask,
                audio_values,
                audio_attention_mask,
                **_embedding_cache_kwargs(batch, device),
            )
            if args.output_head_mode == "legacy":
                out_intensity, out_polarity, _ = model_out
                polarity_idx = torch.argmax(out_polarity, dim=1)
                polarity_preds.extend(polarity_idx.cpu().numpy())
                polarity_labels.extend(polarity.cpu().numpy())
                intensity_preds.extend(out_intensity.detach().cpu().numpy().tolist())
                intensity_targets.extend(intensity.detach().cpu().numpy().tolist())
                signs = torch.where(polarity_idx == 0, -torch.ones_like(out_intensity), torch.ones_like(out_intensity))
                scores = (out_intensity.detach() * signs).cpu().numpy().tolist()
                final_scores.extend(scores)
                raw_valences.extend(raw_valence)
                if args.output_path:
                    for rv, pi, idx in zip(raw_valence, out_intensity.detach().cpu().numpy().tolist(), polarity_idx.cpu().numpy().tolist()):
                        sign = _polarity_sign(idx)
                        output_rows.append({
                            "raw_valence": rv,
                            "Predicted_Intensity": pi,
                            "Predicted_Polarity": sign,
                            "Final_Score": pi * sign,
                        })
            else:
                y_reg = model_out["y_reg"]
                cls7_logits = model_out["cls7_logits"]
                centers = get_cls7_centers(device=cls7_logits.device, dtype=cls7_logits.dtype)
                y_reg_eval = y_reg.clamp(-3.0, 3.0) if args.clamp_regression_eval else y_reg
                y_cls_expected = cls7_expected_value(cls7_logits, centers=centers)
                y_final = compute_final_prediction(y_reg_eval, cls7_logits, eta=args.final_pred_eta, centers=centers)
                cls7_idx = torch.argmax(cls7_logits, dim=1)
                cls7_target = continuous_to_cls7_hard(batch["raw_valence"].to(device))
                reg_scores.extend(y_reg_eval.detach().cpu().numpy().tolist())
                cls_expected_scores.extend(y_cls_expected.detach().cpu().numpy().tolist())
                scores = y_final.detach().cpu().numpy().tolist()
                final_scores.extend(scores)
                raw_valences.extend(raw_valence)
                cls7_preds.extend(cls7_idx.detach().cpu().numpy().tolist())
                cls7_labels.extend(cls7_target.detach().cpu().numpy().tolist())
                if args.output_path:
                    for rv, yr, yc, yf, cidx in zip(raw_valence, y_reg_eval.detach().cpu().numpy().tolist(), y_cls_expected.detach().cpu().numpy().tolist(), scores, cls7_idx.cpu().numpy().tolist()):
                        output_rows.append({
                            "raw_valence": rv,
                            "Predicted_Reg": yr,
                            "Predicted_ClsExpected": yc,
                            "Predicted_Cls7": int(cidx),
                            "Final_Score": yf,
                        })

    if args.output_head_mode == "legacy":
        acc, macro_f1, per_class_precision, per_class_recall = metrics_from_lists(polarity_labels, polarity_preds, num_polarity_classes)
    else:
        acc, macro_f1, per_class_precision, per_class_recall = metrics_from_lists(cls7_labels, cls7_preds, 7)
    
    rmse, mae, pearson, spearman = regression_metrics(final_scores, raw_valences)
    
    pred_7 = [_valence_to_7class(s) for s in final_scores]
    gt_7 = [_valence_to_7class(v) for v in raw_valences]
    acc7, macro_f1_7, _, _ = metrics_from_lists(gt_7, pred_7, 7)
    pred_2 = [_valence_to_binary(s) for s in final_scores]
    gt_2 = [_valence_to_binary(v) for v in raw_valences]
    acc2 = sum(1 for a, b in zip(pred_2, gt_2) if int(a) == int(b)) / max(1, len(gt_2))

    print("\n" + "="*30)
    print("TEST RESULTS")
    print("="*30)
    if args.output_head_mode == "legacy":
        print(f"Polarity Accuracy: {acc:.4f}")
        print(f"Polarity Macro-F1: {macro_f1:.4f}")
    else:
        reg_rmse, reg_mae, _, _ = regression_metrics(reg_scores, raw_valences)
        cls_rmse, cls_mae, _, _ = regression_metrics(cls_expected_scores, raw_valences)
        print(f"Cls7 Accuracy:    {acc:.4f}")
        print(f"Cls7 Macro-F1:    {macro_f1:.4f}")
        print(f"Reg RMSE/MAE:     {reg_rmse:.4f}/{reg_mae:.4f}")
        print(f"ClsExp RMSE/MAE:  {cls_rmse:.4f}/{cls_mae:.4f}")
    print(f"Intensity RMSE:    {rmse:.4f}")
    print(f"Intensity MAE:     {mae:.4f}")
    print(f"Pearson:           {pearson:.4f}")
    print(f"Spearman:          {spearman:.4f}")
    print(f"FinalScore Acc2:   {acc2:.4f}")
    print(f"FinalScore Acc7:   {acc7:.4f}")
    print(f"FinalScore MacroF1:{macro_f1_7:.4f}")
    expert_load_summary = model.get_expert_load_summary(reset=True)
    if expert_load_summary:
        print(f"Expert Load:       {model.expert_load_tracker.brief(expert_load_summary)}")
    print("="*30)

    for i, (p, r) in enumerate(zip(per_class_precision, per_class_recall)):
        print(f"Class {i} Precision: {p:.4f} Recall: {r:.4f}")

    if args.output_path:
        out_dir = os.path.dirname(args.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_path, "w", newline="", encoding="utf-8") as f:
            if args.output_head_mode == "legacy":
                fieldnames = ["raw_valence", "Predicted_Intensity", "Predicted_Polarity", "Final_Score"]
            else:
                fieldnames = ["raw_valence", "Predicted_Reg", "Predicted_ClsExpected", "Predicted_Cls7", "Final_Score"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cmumosei", choices=["cmumosei", "cmumosi"])
    parser.add_argument("--dataset_root", type=str, default="/path/to/user/datasets/MER-unibench/cmumosei-process-complete-20260408")
    parser.add_argument("--embedding_cache_root", type=str, default="")
    parser.add_argument("--tokenizer_path", type=str, default="/path/to/user/models/google-bert_bert-base-uncased")
    parser.add_argument("--vit_model_path", type=str, default="/path/to/user/models/vit-base-patch16-224-in21k")
    parser.add_argument("--bert_model_path", type=str, default="/path/to/user/models/google-bert_bert-base-uncased")
    parser.add_argument("--vision_backbone_type", type=str, default="vit", choices=["resnet18_temporal", "vit"])
    parser.add_argument("--vision_backbone_path", type=str, default="/path/to/user/models/resnet-18")
    parser.add_argument("--bert_backbone", type=str, default="vanilla", choices=["vanilla", "twitter_roberta"])
    parser.add_argument("--vit_backbone", type=str, default="vanilla", choices=["vanilla", "fer"])
    parser.add_argument("--bert_model_path_twitter_roberta", type=str, default="/path/to/user/models/twitter-roberta-base-sentiment-latest")
    parser.add_argument("--vit_model_path_fer", type=str, default="/path/to/user/models/vit-Facial-Expression-Recognition")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--frame_policy", type=str, default="middle")
    parser.add_argument("--hubert_model_path", type=str, default="/path/to/user/models/hubert-base-ls960")
    parser.add_argument("--audio_sample_rate", type=int, default=16000)
    parser.add_argument("--audio_max_seconds", type=float, default=6.0)
    parser.add_argument("--audio_frame_ms", type=float, default=25.0)
    parser.add_argument("--audio_hop_ms", type=float, default=10.0)
    parser.add_argument("--disable_audio_denoise", action="store_true", default=False)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--vit_context_ratio", type=float, default=0.0)
    parser.add_argument("--vit_context_lambda_mode", type=str, default="static", choices=["static", "adaptive", "attention"])
    parser.add_argument("--vit_context_lambda_init", type=str, default="")
    parser.add_argument("--vit_context_attention_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output_head_mode", type=str, default="legacy", choices=["legacy", "signed_reg_cls7"])
    parser.add_argument("--final_pred_eta", type=float, default=0.0)
    parser.add_argument("--clamp_regression_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_experts_msoe", type=int, default=16)
    parser.add_argument("--num_experts_mtoe", type=int, default=32)
    parser.add_argument("--num_shared_experts", type=int, default=2)
    parser.add_argument("--num_text_specific_experts", type=int, default=4)
    parser.add_argument("--num_audio_specific_experts", type=int, default=4)
    parser.add_argument("--num_vision_specific_experts", type=int, default=4)
    parser.add_argument(
        "--enable_text_shared_experts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否允许文本模态访问 shared experts",
    )
    parser.add_argument(
        "--enable_expert_load_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否输出专家负载统计摘要",
    )
    parser.add_argument("--output_path", type=str, default="")
    
    args = parser.parse_args()
    if args.vit_context_attention_dim <= 0:
        raise ValueError("--vit_context_attention_dim must be positive")
    test(args)
