from types import SimpleNamespace
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.emotion_dataset import CMUMOSEIProcessDataset
from head_utils import continuous_to_cls7_hard
from models.models_emotion import EmotionM4OE
from train_emotion import (
    _embedding_cache_kwargs,
    _set_bert_last_layers_requires_grad,
    _set_bert_last_layers_train_mode,
    _set_requires_grad,
    _tcif_context_kwargs,
    _compute_tcif_context_aux_loss,
    _compute_tcif_transition_gate_loss,
)


DATASET_ROOT = os.environ.get("TCIF_DATASET_ROOT", "/path/to/user/datasets/MER-unibench/cmumosei-process-complete-20260408")
CACHE_ROOT = os.environ.get("TCIF_CACHE_ROOT", "/path/to/user/m4oe/embedding_cache/cmumosei_v752_nf4_vctx02_fp16_20260515")
ROBERTA = os.environ.get("TCIF_ROBERTA", "/path/to/user/models/AI-ModelScope_roberta-base")
VIT = os.environ.get("TCIF_VIT", "/path/to/user/models/vit-base-patch16-224-in21k")
HUBERT = os.environ.get("TCIF_HUBERT", "/path/to/user/models/hubert-base-ls960")
RESNET = os.environ.get("TCIF_RESNET", "/path/to/user/models/resnet-18")
SMOKE_BATCH_SIZE = int(os.environ.get("TCIF_SMOKE_BATCH_SIZE", "2"))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("TCIF real-data smoke requires a CUDA allocation")
    device = torch.device("cuda")
    torch.manual_seed(40)
    torch.cuda.manual_seed_all(40)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = CMUMOSEIProcessDataset(
        root=DATASET_ROOT,
        split="train",
        transform=transform,
        tokenizer_path=ROBERTA,
        local_files_only=True,
        frame_policy="middle",
        num_frames=4,
        vit_context_ratio=0.2,
        embedding_cache_root=CACHE_ROOT,
        temporal_context_radius=1,
        temporal_context_mode="neighbors",
        temporal_context_seed=40,
    )
    loader = DataLoader(
        dataset, batch_size=SMOKE_BATCH_SIZE, shuffle=False, num_workers=0
    )
    batch = next(iter(loader))
    valid_context = int(batch["tcif_context_valid_mask"].sum().item())
    if valid_context <= 0:
        raise RuntimeError("Smoke batch unexpectedly has no valid temporal context")

    model = EmotionM4OE(
        num_classes=2,
        num_aux_classes=2,
        embed_dim=768,
        depth_msoe=2,
        depth_mtoe=1,
        num_experts_msoe=16,
        num_experts_mtoe=32,
        num_shared_experts=4,
        num_text_specific_experts=4,
        num_audio_specific_experts=4,
        num_vision_specific_experts=4,
        num_temporal_contrast_experts=0,
        enable_text_shared_experts=False,
        enable_temporal_contrast_experts=False,
        enable_temporal_contrast_loss=True,
        temporal_embedding_dim=128,
        vision_backbone_type="vit",
        vision_backbone_path=RESNET,
        vit_model_path=VIT,
        bert_model_path=ROBERTA,
        hubert_model_path=HUBERT,
        local_files_only=True,
        dropout=0.3,
        router_temperature=0.1,
        enable_expert_load_output=False,
        vit_context_ratio=0.2,
        vit_context_lambda_mode="attention",
        vit_context_lambda_init="0.8,0.1,0.1",
        vit_context_attention_dim=64,
        output_head_mode="signed_reg_cls7",
        final_pred_eta=0.4,
        cls7_head_type="flat",
        enable_tcif=True,
        tcif_latent_dim=128,
        tcif_output_mode="posterior",
        tcif_context_temperature=1.0,
        tcif_enable_transition_gate=True,
        tcif_transition_gate_hidden_dim=64,
        tcif_transition_gate_init_bias=2.0,
    ).to(device)

    _set_requires_grad(model.vit, False)
    _set_requires_grad(model.hubert, False)
    _set_requires_grad(model.bert, False)
    _set_bert_last_layers_requires_grad(model, 2, True)
    model.train()
    model.vit.eval()
    model.hubert.eval()
    model.bert.eval()
    _set_bert_last_layers_train_mode(model, 2, True)

    images = batch["image"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    audio_values = batch["audio_values"].to(device)
    audio_attention_mask = batch["audio_attention_mask"].to(device)
    raw_valence = batch["raw_valence"].to(device)
    output = model(
        images,
        input_ids,
        attention_mask,
        audio_values,
        audio_attention_mask,
        **_embedding_cache_kwargs(batch, device, use_text_cache=False),
        **_tcif_context_kwargs(batch, device, use_text_cache=False),
    )
    args = SimpleNamespace(
        reg_loss_type="smooth_l1",
        tcif_context_aux_cls7_weight=0.5,
        tcif_transition_gate_tau=0.75,
        tcif_transition_gate_conflict_target=0.0,
    )
    context_loss, context_reg, context_cls7, context_count = (
        _compute_tcif_context_aux_loss(output, raw_valence, args)
    )
    transition_gate_loss, transition_gate_stats = (
        _compute_tcif_transition_gate_loss(output, batch, raw_valence, args)
    )
    loss = (
        F.smooth_l1_loss(output["y_reg"], raw_valence)
        + 0.5
        * F.cross_entropy(
            output["cls7_logits"],
            continuous_to_cls7_hard(raw_valence),
        )
        + 0.1 * context_loss
        + 0.05 * transition_gate_loss
    )
    loss.backward()

    projection_grad = model.tcif_regression.output_projection.weight.grad
    context_reg_grad = model.tcif_context_reg_head.weight.grad
    context_cls_grad = model.tcif_context_cls7_head.weight.grad
    transition_gate_grad = model.tcif_regression.transition_gate[-1].weight.grad
    if projection_grad is None or float(projection_grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("TCIF posterior residual projection received no gradient")
    if context_reg_grad is None or float(context_reg_grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("TCIF context regression head received no gradient")
    if context_cls_grad is None or float(context_cls_grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("TCIF context cls7 head received no gradient")
    if transition_gate_grad is None or float(transition_gate_grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("TCIF transition gate received no gradient")

    print(
        {
            "status": "ok",
            "batch_size": int(raw_valence.shape[0]),
            "valid_context_slots": valid_context,
            "context_aux_valid_samples": int(context_count),
            "loss": float(loss.detach().item()),
            "context_loss": float(context_loss.detach().item()),
            "context_reg_loss": float(context_reg.detach().item()),
            "context_cls7_loss": float(context_cls7.detach().item()),
            "transition_gate_loss": float(transition_gate_loss.detach().item()),
            "transition_gate_stats": transition_gate_stats,
            "y_reg_shape": list(output["y_reg"].shape),
            "cls7_shape": list(output["cls7_logits"].shape),
            "max_cuda_memory_gib": float(
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            ),
        }
    )


if __name__ == "__main__":
    main()
