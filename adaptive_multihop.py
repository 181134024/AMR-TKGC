import os
import json
from collections import defaultdict as ddict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from run import Runner
from model import DisenCSPROM
from helper import *
from data_loader import TrainDataset, TestDataset

class ConfidenceEstimator(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.confidence_net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, head_emb: torch.Tensor, rel_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([head_emb, rel_emb], dim=-1)
        conf = self.confidence_net(combined)
        return conf.squeeze(-1)


class DynamicMultiHopPDKGC(DisenCSPROM):
    
    def __init__(self, edge_index, edge_type, params):
        super().__init__(edge_index, edge_type, params=params)
        
       
        self.confidence_estimator = ConfidenceEstimator(
            embed_dim=params.embed_dim,
            hidden_dim=getattr(params, 'confidence_hidden_dim', 128)
        )
        
        self.confidence_threshold = getattr(params, 'confidence_threshold', 0.8)
        self.max_hops = max(1, int(getattr(params, 'max_hops', 3)))
      
        self.hop1_threshold = float(getattr(params, 'hop1_threshold', 0.5))
        self.hop2_threshold = float(getattr(params, 'hop2_threshold', 0.4))

        self.hop3_threshold = float(getattr(params, 'hop3_threshold', 0.3))
        self.hop4_threshold = float(getattr(params, 'hop4_threshold', 0.2))
        

        self.hop_statistics = {h: 0 for h in range(1, self.max_hops + 1)}
        self.total_queries = 0

        self.avg_conf_sum = 0.0
        self.avg_conf_count = 0

        self.conf_min = float('inf')
        self.conf_max = float('-inf')


        self.output_transform: Optional[nn.Linear] = None

    def forward_dynamic_gcn(self, hops: int, mode='train') -> torch.Tensor:

        x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.embed_dim)
        r = self.init_rel
        

        if hops == 0:
            return x
        

        layers_to_run = min(max(0, hops), len(self.conv_ls))
        for i in range(layers_to_run):
            x, r = self.conv_ls[i](x, r, mode)
            if mode == 'train':
                x = self.drop(x)
                
        return x
    
    def _score_one_group(self, sub: torch.Tensor, rel: torch.Tensor, ent_embed: torch.Tensor, text_ids=None, text_mask=None, pred_pos=None) -> tuple:

        sub_emb = torch.index_select(ent_embed, 0, sub)
        rel_emb_single = torch.index_select(self.init_rel, 0, rel)
        rel_emb = rel_emb_single.repeat(1, self.p.num_factors).view(-1, self.p.num_factors, self.p.embed_dim)
        

        if text_ids is None or text_mask is None or pred_pos is None:
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_emb, rel_emb]))
            attention = nn.Softmax(dim=-1)(attention)
            x = self.score_func(sub_emb, rel_emb)
            x = self.score_func.get_logits(x, ent_embed, self.bias)
            pred_logits = torch.einsum('bk,bkn->bn', [attention, x])
            return pred_logits, pred_logits
        


        sub_emb_view = sub_emb.view(-1, self.p.num_factors, self.p.embed_dim)
        rel_emb_exd = torch.unsqueeze(rel_emb_single, 1)
        embed_input = torch.cat([sub_emb_view, rel_emb_exd], dim=1)
        

        prompt = self.prompter(embed_input)
        prompt_attention_mask = torch.ones(sub_emb.size(0), self.p.prompt_length * (self.p.num_factors + 1)).type_as(text_mask)
        text_mask_combined = torch.cat((prompt_attention_mask, text_mask), dim=1)
        

        output = self.plm(input_ids=text_ids, attention_mask=text_mask_combined, layerwise_prompt=prompt)
        last_hidden_state = output.last_hidden_state
        

        prompt_total_len = self.p.prompt_length * (self.p.num_factors + 1)
        ent_rel_state = last_hidden_state[:, :prompt_total_len]
        plm_embeds = torch.chunk(ent_rel_state, chunks=(self.p.num_factors + 1), dim=1)
        plm_sub_embeds, plm_rel_embed = plm_embeds[:self.p.num_factors], plm_embeds[-1]
        
        plm_sub_embed = torch.stack(plm_sub_embeds, dim=1)
        plm_sub_embed = self.llm_fc(plm_sub_embed.reshape(sub_emb.size(0), self.p.num_factors, -1))
        plm_rel_embed = self.llm_fc(plm_rel_embed.reshape(rel_emb.size(0), -1)).repeat(1, self.p.num_factors)
        plm_rel_embed = plm_rel_embed.view(-1, self.p.num_factors, self.p.embed_dim)
        

        attention_plm = self.leakyrelu(torch.einsum('bkf,bkf->bk', [plm_sub_embed, plm_rel_embed]))
        attention_plm = nn.Softmax(dim=-1)(attention_plm)
        x_plm = self.score_func(plm_sub_embed, plm_rel_embed)
        x_plm = self.score_func.get_logits(x_plm, ent_embed, self.bias)
        pred_logits = torch.einsum('bk,bkn->bn', [attention_plm, x_plm])
        

        mask_token_state = []
        for i in range(sub.size(0)):

            pos_idx = pred_pos[i].item() if pred_pos[i].dim() == 0 else pred_pos[i]
            if isinstance(pos_idx, torch.Tensor):
                pos_idx = pos_idx.item()

            pos_idx = pos_idx + prompt_total_len

            seq_len = last_hidden_state.size(1)
            if pos_idx < 0 or pos_idx >= seq_len:
                print(f"[Warning] Offset pred_pos {pos_idx} exceeds sequence length {seq_len}, using last position")
                pos_idx = seq_len - 1
            pred_embed = last_hidden_state[i, pos_idx]
            mask_token_state.append(pred_embed)
        
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
        output_logits = torch.einsum('bf,nf->bn', [output_tmp, self.ent_text_embeds])
        
        return pred_logits, output_logits
    
    def reset_hop_statistics(self):
        self.hop_statistics = {h: 0 for h in range(1, self.max_hops + 1)}
        self.total_queries = 0
        self.avg_conf_sum = 0.0
        self.avg_conf_count = 0
        self.conf_min = float('inf')
        self.conf_max = float('-inf')
    
    def get_hop_statistics(self) -> dict:
        if self.total_queries == 0:
            return {str(h): 0.0 for h in range(1, self.max_hops + 1)}
        
        stats = {}
        for h in range(1, self.max_hops + 1):
            ratio = self.hop_statistics.get(h, 0) / self.total_queries
            stats[str(h)] = ratio
        return stats
    
    def get_hop_counts(self) -> dict:
        counts = {}
        for h in range(1, self.max_hops + 1):
            counts[str(h)] = self.hop_statistics.get(h, 0)
        return counts
    
    def get_confidence_stats(self) -> dict:
        avg_conf = self.avg_conf_sum / max(1, self.avg_conf_count)
        return {
            'min': self.conf_min if self.conf_min != float('inf') else 0.0,
            'max': self.conf_max if self.conf_max != float('-inf') else 0.0,
            'avg': avg_conf
        }

    def estimate_confidence(self, sub: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        sub = sub.long().to(self.init_embed.device)
        rel = rel.long().to(self.init_rel.device)
        head_emb = torch.index_select(self.init_embed, 0, sub)
        rel_emb = torch.index_select(self.init_rel, 0, rel)
        conf = self.confidence_estimator(head_emb, rel_emb)
        return conf

    def compute_confidence_loss(self, confidence: torch.Tensor, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        if targets.dim() == 1:

            target_scores = pred_logits.gather(1, targets.unsqueeze(1)).squeeze(1)

            mask = torch.ones_like(pred_logits, dtype=torch.bool)
            mask.scatter_(1, targets.unsqueeze(1), False)
            max_wrong_scores = pred_logits.masked_fill(~mask, float('-inf')).max(dim=1)[0]
            margins = target_scores - max_wrong_scores
        else:

            target_scores = (pred_logits * targets).sum(dim=1) / targets.sum(dim=1).clamp(min=1)
            wrong_mask = 1 - targets
            max_wrong_scores = (pred_logits * wrong_mask).max(dim=1)[0]
            margins = target_scores - max_wrong_scores
        

        normalized_margins = torch.sigmoid(margins)
        

        confidence_loss = F.mse_loss(confidence, normalized_margins)
        
        return confidence_loss

    def decide_hops(self, confidence: torch.Tensor) -> torch.Tensor:
        hops = torch.ones_like(confidence, dtype=torch.long, device=confidence.device)
        

        conf_min = float(confidence.min().item())
        conf_max = float(confidence.max().item()) 
        conf_mean = float(confidence.mean().item())
        conf_std = float(confidence.std().item())
        







        if self.max_hops >= 5:
            mask5 = confidence < self.hop4_threshold
            hops[mask5] = 5
        if self.max_hops >= 4:
            mask4 = (confidence >= self.hop4_threshold) & (confidence < self.hop3_threshold)
            hops[mask4] = 4
        if self.max_hops >= 3:
            mask3 = (confidence >= self.hop3_threshold) & (confidence < self.hop2_threshold)
            hops[mask3] = 3
        if self.max_hops >= 2:
            mask2 = (confidence >= self.hop2_threshold) & (confidence < self.hop1_threshold)
            hops[mask2] = 2

        hops = torch.clamp(hops, min=1, max=self.max_hops)
        

        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 100 == 1:
            print(f"[Hop Decision Debug] Batch size: {confidence.size(0)}")

            conf_sample = confidence.detach().cpu().view(-1)
            sample_count = min(8, conf_sample.numel())
            sample_vals = ", ".join([f"{float(conf_sample[i]):.6f}" for i in range(sample_count)])
            print(f"  Confidence range: [{conf_min:.6f}, {conf_max:.6f}], mean: {conf_mean:.6f}, std: {conf_std:.6f}")
            print(f"  Confidence samples (first {sample_count}): [{sample_vals}]")

            thr_desc = [f"1-hop>={self.hop1_threshold}",
                        f"2-hop∈[{self.hop2_threshold}, {self.hop1_threshold})",
                        f"3-hop∈[{self.hop3_threshold}, {self.hop2_threshold})" if self.max_hops >= 3 else None,
                        f"4-hop∈[{self.hop4_threshold}, {self.hop3_threshold})" if self.max_hops >= 4 else None,
                        f"5-hop<{self.hop4_threshold}" if self.max_hops >= 5 else None]
            thr_desc = [t for t in thr_desc if t is not None]
            print("  Threshold settings: " + "; ".join(thr_desc))

            binc = torch.bincount(hops, minlength=self.max_hops + 1)
            distrib = [int(binc[h].item()) for h in range(1, self.max_hops + 1)]
            print(f"  Actual hops: {distrib}")
        

        self.total_queries += int(confidence.numel())
        self.avg_conf_sum += float(confidence.sum().item())
        self.avg_conf_count += int(confidence.numel())
        self.conf_min = min(self.conf_min, conf_min)
        self.conf_max = max(self.conf_max, conf_max)
        binc = torch.bincount(hops, minlength=self.max_hops+1)
        for h in range(1, self.max_hops+1):
            self.hop_statistics[h] = self.hop_statistics.get(h, 0) + int(binc[h].item())
        
        return hops


    def forward(self, sub, rel, text_ids, text_mask, pred_pos, mode='train'):

        confidence = self.estimate_confidence(sub, rel)
        hops = self.decide_hops(confidence)

        B = sub.size(0)
        device = self.init_embed.device
        num_ent = self.p.num_ent


        pred = torch.zeros(B, num_ent, device=device)
        output = torch.zeros(B, num_ent, device=device)


        for h in range(1, self.max_hops + 1):
            idx = torch.nonzero(hops == h, as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue

            ent_embed_h = self.forward_dynamic_gcn(h, mode=mode)

            sub_h = sub.index_select(0, idx)
            rel_h = rel.index_select(0, idx)

            text_ids_h = text_ids.index_select(0, idx) if text_ids is not None else None
            text_mask_h = text_mask.index_select(0, idx) if text_mask is not None else None
            pred_pos_h = pred_pos.index_select(0, idx) if pred_pos is not None else None

            pred_h, output_h = self._score_one_group(sub_h, rel_h, ent_embed_h, text_ids_h, text_mask_h, pred_pos_h)

            pred.index_copy_(0, idx, pred_h)
            output.index_copy_(0, idx, output_h)

        return pred, output, confidence


class DynamicMultiHopRunner(Runner):
    def __init__(self, params):

        self.confidence_threshold = getattr(params, 'confidence_threshold', 0.8)
        self.max_hops = max(1, int(getattr(params, 'max_hops', 3)))

        self.hop1_threshold = float(getattr(params, 'hop1_threshold', 0.5))
        self.hop2_threshold = float(getattr(params, 'hop2_threshold', 0.4))
        self.hop3_threshold = float(getattr(params, 'hop3_threshold', 0.3))
        self.hop4_threshold = float(getattr(params, 'hop4_threshold', 0.2))

        self.use_amp = bool(getattr(params, 'enable_amp', False) or getattr(params, 'amp', False))
        self.grouped_backward = bool(getattr(params, 'grouped_backward', False))
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_clip_norm = float(getattr(params, 'grad_clip_norm', 0.0))
        super().__init__(params)
    
    def add_model(self, model_name):
        if model_name.lower() == 'dynamic_multihop':
            if self.max_hops >= 5:
                print(f"Creating dynamic multi-hop model, device: {self.device}, thresholds: hop1={self.hop1_threshold}, hop2={self.hop2_threshold}, hop3={self.hop3_threshold}, hop4={self.hop4_threshold}")
            elif self.max_hops >= 3:
                print(f"Creating dynamic multi-hop model, device: {self.device}, thresholds: hop1={self.hop1_threshold}, hop2={self.hop2_threshold}, hop3={self.hop3_threshold}")
            else:
                print(f"Creating dynamic multi-hop model, device: {self.device}, thresholds: hop1={self.hop1_threshold}, hop2={self.hop2_threshold}")
            model = DynamicMultiHopPDKGC(self.edge_index.to(self.device),
                                         self.edge_type.to(self.device),
                                         self.p)
            model.to(self.device)
            print("Dynamic multi-hop model created successfully")
            return model
        elif model_name.lower() == 'disenkgat':
            model = DisenCSPROM(self.edge_index, self.edge_type, params=self.p)
            model.to(self.device)
            return model
        else:
            try:
                return super().add_model(model_name)
            except NotImplementedError:
                model = DisenCSPROM(self.edge_index, self.edge_type, params=self.p)
                model.to(self.device)
                return model

    def run_epoch(self, epoch):
        print(f"Starting training for epoch {epoch}...")
        if hasattr(self.model, 'reset_hop_statistics'):
            self.model.reset_hop_statistics()

        self.model.train()
        losses, losses_struc, losses_lm, corr_losses, conf_losses = [], [], [], [], []

        print("Creating training iterator...")
        train_iter = iter(self.data_iter['train'])
        print(f"Training dataset size: {len(self.data_iter['train'])}")


        for step, batch in enumerate(train_iter):
            try:
                self.optimizer.zero_grad()
                sub, rel, obj, label, text_ids, text_mask, pred_pos = self.read_batch(batch, 'train')


                if self.grouped_backward:
                    with torch.no_grad():
                        conf_tmp = self.model.estimate_confidence(sub, rel)
                        hops = self.model.decide_hops(conf_tmp)
                    B = sub.size(0)
                    batch_loss_sum = 0.0
                    batch_loss_struc_sum = 0.0
                    batch_loss_lm_sum = 0.0
                    batch_conf_loss_sum = 0.0

                    for h in range(1, self.max_hops + 1):
                        idx = torch.nonzero(hops == h, as_tuple=False).view(-1)
                        if idx.numel() == 0:
                            continue
                        scale = float(idx.numel()) / float(B)

                        with autocast(enabled=self.use_amp):

                            sub_h = sub.index_select(0, idx)
                            rel_h = rel.index_select(0, idx)
                            label_h = label.index_select(0, idx) if label.dim() == 2 else label.index_select(0, idx)
                            text_ids_h = text_ids.index_select(0, idx) if text_ids is not None else None
                            text_mask_h = text_mask.index_select(0, idx) if text_mask is not None else None
                            pred_pos_h = pred_pos.index_select(0, idx) if pred_pos is not None else None


                            conf_scores_h = self.model.estimate_confidence(sub_h, rel_h)


                            ent_embed_h = self.model.forward_dynamic_gcn(h, mode='train')
                            pred_h, output_h = self.model._score_one_group(sub_h, rel_h, ent_embed_h, text_ids_h, text_mask_h, pred_pos_h)


                            x_base_h = self.model.forward_dynamic_gcn(0, mode='train')
                            pred_base_h, _ = self.model._score_one_group(sub_h, rel_h, x_base_h)

                            loss_struc_h = self.model.loss_fn(pred_h, label_h)
                            loss_lm_h = self.model.loss_fn(output_h, label_h)
                            confidence_loss_h = self.model.compute_confidence_loss(conf_scores_h, pred_base_h, label_h)

                            confidence_weight = getattr(self.p, 'confidence_loss_weight', 0.0)
                            if hasattr(self.model, 'loss_weight') and self.model.loss_weight:
                                base_loss_h = self.model.loss_weight(loss_struc_h, loss_lm_h) + confidence_weight * confidence_loss_h
                            else:
                                base_loss_h = loss_struc_h + loss_lm_h + confidence_weight * confidence_loss_h
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
                    corr_losses.append(0.0)
                    conf_losses.append(batch_conf_loss_sum)

                else:

                    with autocast(enabled=self.use_amp):
                        pred, output, conf_scores = self.model.forward(sub, rel, text_ids, text_mask, pred_pos, 'train')

                        loss_struc = self.model.loss_fn(pred, label)
                        loss_lm = self.model.loss_fn(output, label)

                        x_base = self.model.forward_dynamic_gcn(0, mode='train')
                        pred_base, _ = self.model._score_one_group(sub, rel, x_base)
                        confidence_loss = self.model.compute_confidence_loss(conf_scores, pred_base, label)

                        confidence_weight = getattr(self.p, 'confidence_loss_weight', 0.0)
                        if hasattr(self.model, 'loss_weight') and self.model.loss_weight:
                            loss = self.model.loss_weight(loss_struc, loss_lm) + confidence_weight * confidence_loss
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
                    corr_losses.append(0.0)
                    conf_losses.append(confidence_loss.item() if isinstance(confidence_loss, torch.Tensor) else float(confidence_loss))


                lld_losses = []
                for _ in range(self.p.mi_epoch):
                    self.model.mi_Discs.train()
                    lld_loss = self.model.lld_best(sub, rel)
                    self.optimizer_mi.zero_grad()
                    lld_loss.backward()
                    self.optimizer_mi.step()
                    lld_losses.append(lld_loss.item())

                if step % 100 == 0:
                    print(f'[E:{epoch}] Train Loss:{np.mean(losses):.5f}, '
                          f'Struc Loss:{np.mean(losses_struc):.5f}, '
                          f'LM Loss:{np.mean(losses_lm):.5f}, '
                          f'Conf Loss:{np.mean(conf_losses):.5f}')

                if step > 0 and step % 500 == 0:
                    print(f"Processed {step+1} batches")

            except Exception as e:
                print(f"Error processing batch {step}: {e}")
                import traceback
                traceback.print_exc()
                break

        print(f"Training loop completed, processed {step+1} batches")
        try:
            if hasattr(self.model, 'get_hop_statistics'):
                hop_stats = self.model.get_hop_statistics()

                avg_hop = sum(((int(h) if not isinstance(h, int) else h) * float(r))
                               for h, r in hop_stats.items()
                               if (isinstance(h, int) or (isinstance(h, str) and h.isdigit())))

                conf_stats = self.model.get_confidence_stats() if hasattr(self.model, 'get_confidence_stats') else {"min": float('nan'), "max": float('nan'), "avg": float('nan')}
                hop_counts = self.model.get_hop_counts() if hasattr(self.model, 'get_hop_counts') else {}
                print(f"Hop ratio statistics: {hop_stats} | Average hop ≈ {avg_hop:.3f}")
                print(f"Confidence statistics: min={conf_stats['min']:.4f}, max={conf_stats['max']:.4f}, avg={conf_stats['avg']:.4f}")
                print(f"Hop sample counts: {hop_counts} | Total samples: {self.model.total_queries}")
        except Exception as e:
            print(f"Error getting hop statistics: {e}")

        return np.mean(losses), np.mean(conf_losses), 0.0

    @torch.no_grad()
    def _to_multihot(self, label: torch.Tensor, num_ent: int, device):
        if label.dim() == 1:
            idx = label.long().to(device)
            mh = torch.zeros(label.size(0), num_ent, device=device)
            mh.scatter_(1, idx.unsqueeze(1), 1.0)
            return mh
        if label.dim() == 2 and label.size(1) == 1:
            idx = label.view(-1).long().to(device)
            mh = torch.zeros(label.size(0), num_ent, device=device)
            mh.scatter_(1, idx.unsqueeze(1), 1.0)
            return mh
        if label.dim() == 2 and label.size(1) == num_ent:
            return label.float().to(device)

        if label.dim() == 2:

            B = label.size(0)
            mh = torch.zeros(B, num_ent, device=device)
            for b in range(B):
                inds = label[b].long().to(device)
                inds = inds[inds >= 0]
                if inds.numel() > 0:
                    mh[b].scatter_(0, inds, 1.0)
            return mh

        return label.float().to(device)

    def evaluate(self, split, epoch):
        print(f"Starting evaluation for {split} split...")
        required_keys = [f'{split}_head', f'{split}_tail']
        for key in required_keys:
            if key not in self.data_iter:
                print(f"Warning: Missing data iterator {key}")
                return {'mrr': 0.0}, {'mrr': 0.0}, {'mrr': 0.0}

        if hasattr(self.model, 'reset_hop_statistics'):
            self.model.reset_hop_statistics()

        try:
            results = super().evaluate(split, epoch)
            if hasattr(self.model, 'get_hop_statistics'):
                hop_stats = self.model.get_hop_statistics()
                avg_hop = sum(((int(h) if not isinstance(h, int) else h) * float(r))
                               for h, r in hop_stats.items()
                               if (isinstance(h, int) or (isinstance(h, str) and h.isdigit())))
                conf_stats = self.model.get_confidence_stats() if hasattr(self.model, 'get_confidence_stats') else {"min": float('nan'), "max": float('nan'), "avg": float('nan')}
                hop_counts = self.model.get_hop_counts() if hasattr(self.model, 'get_hop_counts') else {}
                print(f"Evaluation hop ratio: {hop_stats} | Average hop ≈ {avg_hop:.3f}")
                print(f"Evaluation confidence statistics: min={conf_stats['min']:.4f}, max={conf_stats['max']:.4f}, avg={conf_stats['avg']:.4f}")
                print(f"Evaluation hop sample counts: {hop_counts} | Total samples: {self.model.total_queries}")
            return results
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {'mrr': 0.0}, {'mrr': 0.0}, {'mrr': 0.0}
def create_dynamic_multihop_runner(params):
    params.model = 'dynamic_multihop'
    runner = DynamicMultiHopRunner(params)
    return runner

