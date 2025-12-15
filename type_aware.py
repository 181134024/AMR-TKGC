import torch
import os
from collections import defaultdict as ddict

class TypeAwareModule:
    
    def __init__(self, num_rel, num_ent, device, enable_type_aware=True, prior_mode='add', prior_alpha=0.3, prior_tau=1.0, prior_eps=1e-6,
                 prior_zero_mean=True, mask_mode='soft', mask_penalty=1.0):
        self.num_rel = num_rel
        self.num_ent = num_ent
        self.device = device
        self.enable_type_aware = enable_type_aware
        self.rel_candidate_mask = None
        self.prior_mode = str(prior_mode)
        self.prior_alpha = float(prior_alpha)
        self.prior_tau = float(prior_tau) if prior_tau is not None and prior_tau > 0 else 1.0
        self.prior_eps = float(prior_eps) if prior_eps is not None else 1e-6
        self.prior_zero_mean = bool(prior_zero_mean)
        mm = str(mask_mode).lower() if mask_mode is not None else 'soft'
        self.mask_mode = 'soft' if mm not in ('soft', 'none') else mm
        self.mask_penalty = float(mask_penalty)
        self.rel_prior_probs = None
        self.rel_prior_log = None
        self.rel_candidate_density = None
        
    def build_relation_candidate_mask(self, sr2o):
        if not self.enable_type_aware:
            return None
            
        rel_candidate_mask = torch.zeros(self.num_rel * 2, self.num_ent, dtype=torch.bool)
        
        for (s, r), objs in sr2o.items():
            for o in objs:
                rel_candidate_mask[r, o] = True
        
        candidate_counts = rel_candidate_mask.sum(dim=1).float()
        density = candidate_counts / max(1, self.num_ent)
        self.rel_candidate_density = density.to(self.device)

        self.rel_candidate_mask = rel_candidate_mask.to(self.device)
        return self.rel_candidate_mask

    def build_relation_prior(self, sr2o):
        if not self.enable_type_aware:
            return None
        counts = torch.zeros(self.num_rel * 2, self.num_ent, dtype=torch.float32)
        for (s, r), objs in sr2o.items():
            for o in objs:
                counts[r, o] += 1.0
        probs = counts + self.prior_eps
        denom = probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        probs = probs / denom
        log_prior = torch.log(probs.clamp_min(1e-12)) / self.prior_tau
        self.rel_prior_probs = probs.to(self.device)
        self.rel_prior_log = log_prior.to(self.device)
        return self.rel_prior_probs

    def apply_type_aware_mask(self, pred_scores, output_scores, rel, obj):
        if not self.enable_type_aware or self.rel_candidate_mask is None:
            return pred_scores, output_scores, {'covered': 0.0, 'total': 0.0, 'cand_total': 0.0}
        b_range = torch.arange(pred_scores.size()[0], device=self.device)
        batch_rel_mask = self.rel_candidate_mask[rel]
        if batch_rel_mask.dtype != torch.bool:
            batch_rel_mask = batch_rel_mask.bool()
        type_aware_stats = {
            'covered': batch_rel_mask[b_range, obj].float().sum().item(),
            'total': obj.size(0),
            'cand_total': batch_rel_mask.float().sum().item()
        }
        penalty = (~batch_rel_mask).float() * self.mask_penalty
        masked_pred_scores = pred_scores - penalty
        masked_output_scores = output_scores - penalty
        return masked_pred_scores, masked_output_scores, type_aware_stats

    def apply_type_prior_bias(self, pred_scores, output_scores, rel, obj=None):
        if not self.enable_type_aware or self.rel_prior_log is None:
            return pred_scores, output_scores, {'covered': 0.0, 'total': float(obj.size(0)) if obj is not None else 0.0, 'cand_total': 0.0}
        if self.prior_mode == 'mul':
            bias = self.prior_alpha * torch.log(self.rel_prior_probs[rel].clamp_min(1e-12))
        else:
            bias = self.prior_alpha * self.rel_prior_log[rel]
        if self.prior_zero_mean:
            bias = bias - bias.mean(dim=1, keepdim=True)
        biased_pred = pred_scores + bias
        biased_output = output_scores + bias
        if self.rel_candidate_mask is not None and obj is not None:
            b_range = torch.arange(biased_pred.size()[0], device=self.device)
            batch_rel_mask = self.rel_candidate_mask[rel]
            if batch_rel_mask.dtype != torch.bool:
                batch_rel_mask = batch_rel_mask.bool()
            type_aware_stats = {
                'covered': batch_rel_mask[b_range, obj].float().sum().item(),
                'total': float(obj.size(0)),
                'cand_total': batch_rel_mask.float().sum().item()
            }
        else:
            type_aware_stats = {'covered': 0.0, 'total': float(obj.size(0)) if obj is not None else 0.0, 'cand_total': 0.0}
        return biased_pred, biased_output, type_aware_stats

    def apply_type_aware(self, pred_scores, output_scores, rel, obj=None):
        if not self.enable_type_aware:
            return pred_scores, output_scores, {'covered': 0.0, 'total': float(obj.size(0)) if obj is not None else 0.0, 'cand_total': 0.0}
        scores_pred, scores_output, _ = self.apply_type_prior_bias(pred_scores, output_scores, rel, obj)
        if self.rel_candidate_mask is None or self.mask_mode == 'none':
            stats = {'covered': 0.0, 'total': float(obj.size(0)) if obj is not None else 0.0, 'cand_total': 0.0}
            return scores_pred, scores_output, stats
        b_range = torch.arange(scores_pred.size(0), device=self.device)
        batch_rel_mask = self.rel_candidate_mask[rel]
        if batch_rel_mask.dtype != torch.bool:
            batch_rel_mask = batch_rel_mask.bool()
        if obj is not None:
            covered = batch_rel_mask[b_range, obj].float().sum().item()
            total = float(obj.size(0))
        else:
            covered, total = 0.0, 0.0
        cand_total = batch_rel_mask.float().sum().item()
        stats = {'covered': covered, 'total': total, 'cand_total': cand_total}
        if self.mask_mode == 'soft':
            penalty = (~batch_rel_mask).float() * self.mask_penalty
            scores_pred = scores_pred - penalty
            scores_output = scores_output - penalty
        return scores_pred, scores_output, stats

    def get_coverage_stats(self, stats_list):
        total_covered = sum(stats['covered'] for stats in stats_list)
        total_samples = sum(stats['total'] for stats in stats_list)
        total_candidates = sum(stats['cand_total'] for stats in stats_list)
        coverage_rate = (total_covered / total_samples) if total_samples > 0 else 0.0
        avg_candidate_size = (total_candidates / total_samples) if total_samples > 0 else 0.0
        avg_density = avg_candidate_size / float(self.num_ent) if self.num_ent > 0 else 0.0
        return {
            'coverage_rate': coverage_rate,
            'avg_candidate_size': avg_candidate_size,
            'avg_density': avg_density,
            'total_covered': total_covered,
            'total_samples': total_samples
        }
    
    def print_type_aware_report(self, coverage_stats, type_aware_results, baseline_results):
        if not self.enable_type_aware:
            return
            
        print('[Type-Aware Effect Report]')
        print('  Coverage: {:.2%} ({} / {})'.format(
            coverage_stats['coverage_rate'], 
            int(coverage_stats['total_covered']), 
            int(coverage_stats['total_samples'])
        ))
        print('  Avg Candidate Size: {:.2f} / {} (density {:.2%})'.format(
            coverage_stats['avg_candidate_size'], 
            self.num_ent, 
            coverage_stats['avg_density']
        ))
        
        if baseline_results:
            print('  Baseline (No Type-Aware) Results:')
            for key, results in baseline_results.items():
                if 'count' in results:
                    count = results['count']
                    mr = results['mr'] / count
                    mrr = results['mrr'] / count
                    hits1 = results.get('hits@1', 0.0) / count
                    hits3 = results.get('hits@3', 0.0) / count
                    hits10 = results.get('hits@10', 0.0) / count
                    print(f'    {key}: MR {mr:.2f}, MRR {mrr:.4f}, H@1 {hits1:.4f}, H@3 {hits3:.4f}, H@10 {hits10:.4f}')
