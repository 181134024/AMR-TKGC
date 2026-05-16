from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from run import Runner
from model import DisenCSPROM
from helper import *


class ConfidenceEstimator(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        steepness: float = 10.0,
        target_logit_std: float = 1.0,
        max_adaptive_gain: float = 3.0,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(embed_dim * 2)
        self.confidence_net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.steepness = float(steepness)
        self.target_logit_std = float(target_logit_std)
        self.max_adaptive_gain = float(max_adaptive_gain)

    def forward(
        self,
        head_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        combined = torch.cat([head_emb, rel_emb], dim=-1)
        combined = self.input_norm(combined)
        logits = self.confidence_net(combined).squeeze(-1)

        if logits.numel() > 1:
            centered = logits - logits.mean()
            logits_std = centered.std(unbiased=False).clamp(min=1e-6)
            adaptive_gain = torch.clamp(
                self.target_logit_std / logits_std,
                min=1.0,
                max=self.max_adaptive_gain,
            )
            logits = centered * adaptive_gain

        temperature = max(1e-6, float(temperature))
        return torch.sigmoid(logits * (self.steepness / temperature))


class AdaptiveMultiHopPDKGC(DisenCSPROM):
    def __init__(self, edge_index, edge_type, params):
        super().__init__(edge_index, edge_type, params=params)
        self.confidence_estimator = ConfidenceEstimator(
            embed_dim=params.embed_dim,
            hidden_dim=getattr(params, "confidence_hidden_dim", 128),
            steepness=getattr(params, "confidence_steepness", 10.0),
            target_logit_std=getattr(params, "confidence_target_logit_std", 1.0),
            max_adaptive_gain=getattr(params, "confidence_max_gain", 3.0),
        )
        self.max_hops = max(1, int(getattr(params, "max_hops", 3)))
        self.hop1_threshold = float(getattr(params, "hop1_threshold", 0.5))
        self.hop2_threshold = float(getattr(params, "hop2_threshold", 0.4))
        self.hop3_threshold = float(getattr(params, "hop3_threshold", 0.3))
        self.hop4_threshold = float(getattr(params, "hop4_threshold", 0.2))
        self.confidence_temp = float(getattr(params, "confidence_temp", 1.0))
        self.use_percentile_threshold = bool(
            getattr(params, "use_percentile_threshold", False)
        )
        self.output_transform: Optional[nn.Linear] = None

    def forward_adaptive_gcn(self, hops: int, mode: str = "train") -> torch.Tensor:
        x = self.act(self.pca(self.init_embed)).view(
            -1, self.p.num_factors, self.p.embed_dim
        )
        r = self.init_rel

        if hops == 0:
            return x

        for i in range(min(max(0, hops), len(self.conv_ls))):
            x, r = self.conv_ls[i](x, r, mode)
            if mode == "train":
                x = self.drop(x)
        return x

    def _score_one_group(
        self,
        sub: torch.Tensor,
        rel: torch.Tensor,
        ent_embed: torch.Tensor,
        text_ids=None,
        text_mask=None,
        pred_pos=None,
    ) -> tuple:
        sub_emb = torch.index_select(ent_embed, 0, sub)
        rel_emb_single = torch.index_select(self.init_rel, 0, rel)
        rel_emb = rel_emb_single.repeat(1, self.p.num_factors).view(
            -1, self.p.num_factors, self.p.embed_dim
        )

        if text_ids is None or text_mask is None or pred_pos is None:
            attention = self.leakyrelu(torch.einsum("bkf,bkf->bk", [sub_emb, rel_emb]))
            attention = nn.Softmax(dim=-1)(attention)
            x = self.score_func(sub_emb, rel_emb)
            x = self.score_func.get_logits(x, ent_embed, self.bias)
            pred_logits = torch.einsum("bk,bkn->bn", [attention, x])
            return pred_logits, pred_logits

        embed_input = torch.cat(
            [sub_emb.view(-1, self.p.num_factors, self.p.embed_dim), rel_emb_single.unsqueeze(1)],
            dim=1,
        )
        prompt = self.prompter(embed_input)
        prompt_total_len = self.p.prompt_length * (self.p.num_factors + 1)
        prompt_attention_mask = torch.ones(sub_emb.size(0), prompt_total_len).type_as(text_mask)
        text_mask_combined = torch.cat((prompt_attention_mask, text_mask), dim=1)

        output = self.plm(
            input_ids=text_ids,
            attention_mask=text_mask_combined,
            layerwise_prompt=prompt,
        )
        last_hidden_state = output.last_hidden_state

        ent_rel_state = last_hidden_state[:, :prompt_total_len]
        plm_embeds = torch.chunk(ent_rel_state, chunks=self.p.num_factors + 1, dim=1)
        plm_sub_embeds, plm_rel_embed = plm_embeds[: self.p.num_factors], plm_embeds[-1]

        plm_sub_embed = self.llm_fc(
            torch.stack(plm_sub_embeds, dim=1).reshape(
                sub_emb.size(0), self.p.num_factors, -1
            )
        )
        plm_rel_embed = self.llm_fc(plm_rel_embed.reshape(rel_emb.size(0), -1))
        plm_rel_embed = plm_rel_embed.repeat(1, self.p.num_factors).view(
            -1, self.p.num_factors, self.p.embed_dim
        )

        attention_plm = self.leakyrelu(
            torch.einsum("bkf,bkf->bk", [plm_sub_embed, plm_rel_embed])
        )
        attention_plm = nn.Softmax(dim=-1)(attention_plm)
        x_plm = self.score_func(plm_sub_embed, plm_rel_embed)
        x_plm = self.score_func.get_logits(x_plm, ent_embed, self.bias)
        pred_logits = torch.einsum("bk,bkn->bn", [attention_plm, x_plm])

        mask_token_state = []
        for i in range(sub.size(0)):
            pos_idx = pred_pos[i].item() if pred_pos[i].dim() == 0 else pred_pos[i]
            if isinstance(pos_idx, torch.Tensor):
                pos_idx = pos_idx.item()
            pos_idx = min(max(pos_idx + prompt_total_len, 0), last_hidden_state.size(1) - 1)
            mask_token_state.append(last_hidden_state[i, pos_idx])

        if mask_token_state:
            mask_token_state = torch.stack(mask_token_state, dim=0)
        else:
            token_mask = text_mask_combined[:, prompt_total_len:]
            if token_mask.dim() == 2:
                token_mask = token_mask.unsqueeze(-1)
            masked = last_hidden_state[:, prompt_total_len:, :] * token_mask
            denom = token_mask.sum(dim=1).clamp(min=1e-6)
            mask_token_state = masked.sum(dim=1) / denom

        output_tmp = self.ent_transform(mask_token_state)
        output_logits = torch.einsum("bf,nf->bn", [output_tmp, self.ent_text_embeds])
        return pred_logits, output_logits

    def estimate_confidence(self, sub: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        sub = sub.long().to(self.init_embed.device)
        rel = rel.long().to(self.init_rel.device)
        head_emb = torch.index_select(self.init_embed, 0, sub)
        rel_emb = torch.index_select(self.init_rel, 0, rel)
        return self.confidence_estimator(
            head_emb,
            rel_emb,
            temperature=self.confidence_temp,
        )

    def compute_confidence_loss(
        self,
        confidence: torch.Tensor,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if targets.dim() == 1:
            target_scores = pred_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
            mask = torch.ones_like(pred_logits, dtype=torch.bool)
            mask.scatter_(1, targets.unsqueeze(1), False)
            max_wrong_scores = pred_logits.masked_fill(~mask, float("-inf")).max(dim=1)[0]
            margins = target_scores - max_wrong_scores
        else:
            target_scores = (pred_logits * targets).sum(dim=1) / targets.sum(dim=1).clamp(min=1)
            wrong_mask = 1 - targets
            max_wrong_scores = (pred_logits * wrong_mask).max(dim=1)[0]
            margins = target_scores - max_wrong_scores

        normalized_margins = torch.sigmoid(margins / max(1e-6, self.confidence_temp))
        mse_loss = F.mse_loss(confidence, normalized_margins)

        ranking_loss = torch.tensor(0.0, device=confidence.device)
        ranking_weight = getattr(self.p, "confidence_ranking_weight", 0.0)
        if ranking_weight > 0 and confidence.size(0) > 1:
            idx = torch.randperm(confidence.size(0), device=confidence.device)
            conf_perm = confidence.index_select(0, idx)
            margin_perm = margins.index_select(0, idx)
            rank_label = torch.sign(margins - margin_perm)
            rank_label = torch.where(rank_label == 0, torch.ones_like(rank_label), rank_label)
            ranking_loss = F.margin_ranking_loss(
                confidence,
                conf_perm,
                rank_label,
                margin=0.1,
            )

        variance_loss = torch.tensor(0.0, device=confidence.device)
        if confidence.size(0) > 1:
            target_std = float(getattr(self.p, "confidence_target_std", 0.25))
            variance_loss = F.relu(target_std - confidence.std())

        variance_weight = float(getattr(self.p, "confidence_variance_weight", 0.2))
        return mse_loss + ranking_weight * ranking_loss + variance_weight * variance_loss

    def decide_hops(self, confidence: torch.Tensor) -> torch.Tensor:
        hops = torch.ones_like(confidence, dtype=torch.long, device=confidence.device)
        batch_size = confidence.size(0)

        if self.use_percentile_threshold and batch_size > self.max_hops:
            sorted_conf, _ = torch.sort(confidence, descending=True)
            if self.max_hops == 3:
                idx1 = int(batch_size * 0.33)
                idx2 = int(batch_size * 0.66)
                t1 = sorted_conf[idx1].item() if idx1 < batch_size else 0.0
                t2 = sorted_conf[idx2].item() if idx2 < batch_size else 0.0
                hops[(confidence < t1) & (confidence >= t2)] = 2
                hops[confidence < t2] = 3
            elif self.max_hops == 2:
                idx1 = int(batch_size * 0.5)
                t1 = sorted_conf[idx1].item() if idx1 < batch_size else 0.0
                hops[confidence < t1] = 2
            else:
                for h in range(2, self.max_hops + 1):
                    idx = int(batch_size * ((h - 1) / self.max_hops))
                    threshold = sorted_conf[idx].item() if idx < batch_size else 0.0
                    hops[confidence < threshold] = h
        else:
            if self.max_hops >= 5:
                hops[confidence < self.hop4_threshold] = 5
            if self.max_hops >= 4:
                hops[(confidence >= self.hop4_threshold) & (confidence < self.hop3_threshold)] = 4
            if self.max_hops >= 3:
                hops[(confidence >= self.hop3_threshold) & (confidence < self.hop2_threshold)] = 3
            if self.max_hops >= 2:
                hops[(confidence >= self.hop2_threshold) & (confidence < self.hop1_threshold)] = 2

        return torch.clamp(hops, min=1, max=self.max_hops)

    def forward(self, sub, rel, text_ids, text_mask, pred_pos, mode: str = "train"):
        confidence = self.estimate_confidence(sub, rel)
        hops = self.decide_hops(confidence)
        batch_size = sub.size(0)
        device = self.init_embed.device
        num_ent = self.p.num_ent

        pred = torch.zeros(batch_size, num_ent, device=device)
        output = torch.zeros(batch_size, num_ent, device=device)

        for h in range(1, self.max_hops + 1):
            idx = torch.nonzero(hops == h, as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            ent_embed_h = self.forward_adaptive_gcn(h, mode=mode)
            sub_h = sub.index_select(0, idx)
            rel_h = rel.index_select(0, idx)
            text_ids_h = text_ids.index_select(0, idx) if text_ids is not None else None
            text_mask_h = text_mask.index_select(0, idx) if text_mask is not None else None
            pred_pos_h = pred_pos.index_select(0, idx) if pred_pos is not None else None
            pred_h, output_h = self._score_one_group(
                sub_h,
                rel_h,
                ent_embed_h,
                text_ids_h,
                text_mask_h,
                pred_pos_h,
            )
            pred.index_copy_(0, idx, pred_h)
            output.index_copy_(0, idx, output_h)

        return pred, output, confidence


class AdaptiveMultiHopRunner(Runner):
    def __init__(self, params):
        self.max_hops = max(1, int(getattr(params, "max_hops", 3)))
        self.use_amp = bool(getattr(params, "enable_amp", False) or getattr(params, "amp", False))
        self.grouped_backward = bool(getattr(params, "grouped_backward", False))
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_clip_norm = float(getattr(params, "grad_clip_norm", 0.0))
        super().__init__(params)

    def add_model(self, model_name):
        if model_name.lower() == "adaptive_multihop":
            model = AdaptiveMultiHopPDKGC(
                self.edge_index.to(self.device),
                self.edge_type.to(self.device),
                self.p,
            )
            model.to(self.device)
            return model

        if model_name.lower() == "disenkgat":
            model = DisenCSPROM(self.edge_index, self.edge_type, params=self.p)
            model.to(self.device)
            return model

        try:
            return super().add_model(model_name)
        except NotImplementedError:
            model = DisenCSPROM(self.edge_index, self.edge_type, params=self.p)
            model.to(self.device)
            return model

    def run_epoch(self, epoch):
        self.model.train()
        losses, losses_struc, losses_lm, conf_losses = [], [], [], []
        train_iter = iter(self.data_iter["train"])
        max_train_steps = int(getattr(self.p, "max_train_steps", -1))

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            sub, rel, _, label, text_ids, text_mask, pred_pos = self.read_batch(batch, "train")

            if self.grouped_backward:
                with torch.no_grad():
                    hops = self.model.decide_hops(self.model.estimate_confidence(sub, rel))

                batch_size = sub.size(0)
                batch_loss_sum = 0.0
                batch_loss_struc_sum = 0.0
                batch_loss_lm_sum = 0.0
                batch_conf_loss_sum = 0.0

                for h in range(1, self.max_hops + 1):
                    idx = torch.nonzero(hops == h, as_tuple=False).view(-1)
                    if idx.numel() == 0:
                        continue

                    scale = float(idx.numel()) / float(batch_size)
                    with autocast(enabled=self.use_amp):
                        sub_h = sub.index_select(0, idx)
                        rel_h = rel.index_select(0, idx)
                        label_h = label.index_select(0, idx)
                        text_ids_h = text_ids.index_select(0, idx) if text_ids is not None else None
                        text_mask_h = text_mask.index_select(0, idx) if text_mask is not None else None
                        pred_pos_h = pred_pos.index_select(0, idx) if pred_pos is not None else None

                        conf_scores_h = self.model.estimate_confidence(sub_h, rel_h)
                        ent_embed_h = self.model.forward_adaptive_gcn(h, mode="train")
                        pred_h, output_h = self.model._score_one_group(
                            sub_h,
                            rel_h,
                            ent_embed_h,
                            text_ids_h,
                            text_mask_h,
                            pred_pos_h,
                        )
                        x_base_h = self.model.forward_adaptive_gcn(0, mode="train")
                        pred_base_h, _ = self.model._score_one_group(sub_h, rel_h, x_base_h)

                        loss_struc_h = self.model.loss_fn(pred_h, label_h)
                        loss_lm_h = self.model.loss_fn(output_h, label_h)
                        confidence_loss_h = self.model.compute_confidence_loss(
                            conf_scores_h,
                            pred_base_h,
                            label_h,
                        )

                        confidence_weight = getattr(self.p, "confidence_loss_weight", 0.0)
                        if hasattr(self.model, "loss_weight") and self.model.loss_weight:
                            base_loss_h = (
                                self.model.loss_weight(loss_struc_h, loss_lm_h)
                                + confidence_weight * confidence_loss_h
                            )
                        else:
                            base_loss_h = (
                                loss_struc_h + loss_lm_h + confidence_weight * confidence_loss_h
                            )
                        scaled_loss_h = base_loss_h * scale

                    if self.use_amp:
                        self.scaler.scale(scaled_loss_h).backward()
                    else:
                        scaled_loss_h.backward()

                    batch_loss_sum += float(base_loss_h.detach().cpu()) * scale
                    batch_loss_struc_sum += float(loss_struc_h.detach().cpu()) * scale
                    batch_loss_lm_sum += float(loss_lm_h.detach().cpu()) * scale
                    batch_conf_loss_sum += float(confidence_loss_h.detach().cpu()) * scale

                if self.grad_clip_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                losses.append(batch_loss_sum)
                losses_struc.append(batch_loss_struc_sum)
                losses_lm.append(batch_loss_lm_sum)
                conf_losses.append(batch_conf_loss_sum)
            else:
                with autocast(enabled=self.use_amp):
                    pred, output, conf_scores = self.model.forward(
                        sub,
                        rel,
                        text_ids,
                        text_mask,
                        pred_pos,
                        "train",
                    )
                    loss_struc = self.model.loss_fn(pred, label)
                    loss_lm = self.model.loss_fn(output, label)
                    x_base = self.model.forward_adaptive_gcn(0, mode="train")
                    pred_base, _ = self.model._score_one_group(sub, rel, x_base)
                    confidence_loss = self.model.compute_confidence_loss(
                        conf_scores,
                        pred_base,
                        label,
                    )

                    confidence_weight = getattr(self.p, "confidence_loss_weight", 0.0)
                    if hasattr(self.model, "loss_weight") and self.model.loss_weight:
                        loss = (
                            self.model.loss_weight(loss_struc, loss_lm)
                            + confidence_weight * confidence_loss
                        )
                    else:
                        loss = loss_struc + loss_lm + confidence_weight * confidence_loss

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

                losses.append(loss.item())
                losses_struc.append(loss_struc.item())
                losses_lm.append(loss_lm.item())
                conf_losses.append(
                    confidence_loss.item()
                    if isinstance(confidence_loss, torch.Tensor)
                    else float(confidence_loss)
                )

            for _ in range(self.p.mi_epoch):
                self.model.mi_Discs.train()
                lld_loss = self.model.lld_best(sub, rel)
                self.optimizer_mi.zero_grad()
                lld_loss.backward()
                self.optimizer_mi.step()

            if max_train_steps > 0 and step + 1 >= max_train_steps:
                break

        return np.mean(losses), np.mean(conf_losses), 0.0

    def evaluate(self, split, epoch):
        return super().evaluate(split, epoch)



def create_adaptive_multihop_runner(params):
    params.model = "adaptive_multihop"
    return AdaptiveMultiHopRunner(params)
