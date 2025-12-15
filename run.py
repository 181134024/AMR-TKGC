from helper import *
from data_loader import *
from model import *
from type_aware import TypeAwareModule
import transformers
from transformers import AutoConfig, BertTokenizer, RobertaTokenizer
transformers.logging.set_verbosity_error()
from tqdm import tqdm
import traceback
import os


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True



class Runner(object):

    def load_data(self):
        def construct_input_text(sub, rel):
            sub_name = self.ent_names[sub]
            sub_desc = self.ent_descs[sub]
            sub_text = sub_name + ' ' + sub_desc
            if rel < self.p.num_rel:
                rel_text = self.rel_names[rel]
            else:
                rel_text = 'reversed: ' + self.rel_names[rel - self.p.num_rel]
            tokenized_text = self.tok(sub_text, text_pair=rel_text, max_length=self.p.text_len-1, truncation=True)
            source_ids = tokenized_text.input_ids
            source_mask = tokenized_text.attention_mask
            source_ids.insert(-1, self.mask_token_id)
            source_mask.insert(-1, 1)
            return source_ids, source_mask
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, 'data', self.p.dataset)
        fallback_dir = os.path.join(base_dir, 'data', 'ProcKG')
        use_fallback = not os.path.isdir(data_dir)
        data_path = data_dir if not use_fallback else fallback_dir
        name_base = os.path.join(base_dir, 'data') if not use_fallback else os.path.join(base_dir, 'data')
        dataset_key = self.p.dataset if not use_fallback else ''
        ent_id2name = read_file(name_base, dataset_key, 'entityid2name.txt', 'name')
        rel_id2name = read_file(name_base, dataset_key, 'relationid2name.txt', 'name')

        self.ent2id = {name.lower(): idx for idx, name in enumerate(ent_id2name)}
        self.rel2id = {name.lower(): idx for idx, name in enumerate(rel_id2name)}
        self.rel2id.update({(name.lower() + '_reverse'): idx + len(self.rel2id) for idx, name in enumerate(rel_id2name)})

        self.id2ent = {idx: name for idx, name in enumerate(ent_id2name)}
        self.id2rel = {idx: name for idx, name in enumerate(rel_id2name)}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim
        print('num_ent {} num_rel {}'.format(self.p.num_ent, self.p.num_rel))
        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            id_path = os.path.join(data_path, f'{split}2id.txt')
            txt_path = os.path.join(data_path, f'{split}.txt')
            name_path = os.path.join(data_path, f'{split}2id_name.txt')
            
            loaded = False
            # First try ID format (space or tab separated)
            if os.path.exists(id_path):
                lines = open(id_path, encoding='utf-8')
                for line in lines:
                    line = line.strip()
                    if not line or line.isdigit():  # Skip header line (count)
                        continue
                    # Try space-separated first, then tab-separated
                    if ' ' in line:
                        parts = line.split()
                    else:
                        parts = line.split('\t')
                    if len(parts) != 3:
                        continue
                    try:
                        sub_id, obj_id, rel_id = int(parts[0]), int(parts[1]), int(parts[2])
                        # Validate IDs are within range
                        if 0 <= sub_id < self.p.num_ent and 0 <= obj_id < self.p.num_ent and 0 <= rel_id < self.p.num_rel:
                            self.data[split].append((sub_id, rel_id, obj_id))
                            loaded = True
                    except ValueError:
                        continue
            # Then try name format (tab-separated)
            elif os.path.exists(txt_path):
                lines = open(txt_path, encoding='utf-8')
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) != 3:
                        continue
                    sub, rel, obj = map(str.lower, parts)
                    if sub not in self.ent2id or rel not in self.rel2id or obj not in self.ent2id:
                        continue
                    self.data[split].append((self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]))
                    loaded = True
            # Finally try name format with pipe separator
            elif os.path.exists(name_path):
                lines = open(name_path, encoding='utf-8')
                for line in lines:
                    if not line.strip() or ('|' not in line):
                        continue
                    parts = [p.strip().lower() for p in line.strip().split('|')]
                    if len(parts) != 3:
                        continue
                    sub, obj, rel = parts[0], parts[1], parts[2]
                    if sub in self.ent2id and rel in self.rel2id and obj in self.ent2id:
                        self.data[split].append((self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]))
                        loaded = True
            # Ensure split key exists even if no data was loaded
            if split not in self.data:
                self.data[split] = []
            if loaded:
                print(f"[Data Loading] Loaded {len(self.data[split])} {split} samples from {id_path if os.path.exists(id_path) else (txt_path if os.path.exists(txt_path) else name_path)}")



        self.data = dict(self.data)
        # Print data loading statistics
        print(f"[Data Loading] Loaded data counts: train={len(self.data.get('train', []))}, "
              f"test={len(self.data.get('test', []))}, valid={len(self.data.get('valid', []))}")
        
        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for sub, rel, obj in self.data.get('train', []):
            sr2o[(sub, rel)].add(obj)
            sr2o[(obj, rel + self.p.num_rel)].add(sub)
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data.get(split, []):
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

        self.ent_names = read_file(name_base, dataset_key, 'entityid2name.txt', 'name')
        self.rel_names = read_file(name_base, dataset_key, 'relationid2name.txt', 'name')
        self.ent_descs = read_file(name_base, dataset_key, 'entityid2description.txt', 'desc')
        triples_save_file_name = 'bert'
        if not getattr(self.p, 'disable_plm', False):
            if self.p.pretrained_model_name.lower() == 'bert_base' or self.p.pretrained_model_name.lower() == 'bert_large':
                self.tok = BertTokenizer.from_pretrained(self.p.pretrained_model, add_prefix_space=False)
                triples_save_file_name = 'bert'
            elif self.p.pretrained_model_name.lower() == 'roberta_base' or self.p.pretrained_model_name.lower() == 'roberta_large':
                self.tok = RobertaTokenizer.from_pretrained(self.p.pretrained_model, add_prefix_space=True)
                triples_save_file_name = 'roberta'

        triples_save_root = os.path.join(base_dir, 'data', self.p.dataset)
        if not os.path.isdir(triples_save_root):
            triples_save_root = os.path.join(base_dir, 'data', 'ProcKG')
        triples_save_file = os.path.join(triples_save_root, '{}_{}.txt'.format('loaded_triples', triples_save_file_name))

        if os.path.exists(triples_save_file):
            self.triples = json.load(open(triples_save_file, encoding='utf-8'))
            # Ensure all required keys exist
            if 'train' not in self.triples:
                self.triples['train'] = []
            for split in ['test', 'valid']:
                for suffix in ['tail', 'head']:
                    key = '{}_{}'.format(split, suffix)
                    if key not in self.triples:
                        self.triples[key] = []
            # Check if training data is empty
            if len(self.triples.get('train', [])) == 0:
                print(f"[Warning] Loaded triples file exists but 'train' is empty. Regenerating triples...")
                # Delete the empty file and regenerate
                os.remove(triples_save_file)
                self.triples = ddict(list)
            else:
                print(f"[Info] Loaded triples from file: train={len(self.triples.get('train', []))} samples")
        
        if not os.path.exists(triples_save_file) or len(self.triples.get('train', [])) == 0:
            self.triples = ddict(list)
            for sub, rel, obj in tqdm(self.data.get('train', [])):
                if getattr(self.p, 'disable_plm', False):
                    text_ids, text_mask, pred_pos = [], [], 0
                else:
                    text_ids, text_mask = construct_input_text(sub, rel)
                    pred_pos = text_ids.index(self.mask_token_id)
                self.triples['train'].append({'triple': (sub, rel, -1), 'label': [obj], 'sub_samp': 1, 'text_ids': text_ids, 'text_mask': text_mask, 'pred_pos': pred_pos})
                rel_inv = rel + self.p.num_rel
                if getattr(self.p, 'disable_plm', False):
                    text_ids, text_mask, pred_pos = [], [], 0
                else:
                    text_ids, text_mask = construct_input_text(obj, rel_inv)
                    pred_pos = text_ids.index(self.mask_token_id)
                self.triples['train'].append({'triple': (obj, rel_inv, -1), 'label': [sub], 'sub_samp': 1, 'text_ids': text_ids, 'text_mask': text_mask, 'pred_pos': pred_pos})

            for split in ['test', 'valid']:
                for sub, rel, obj in tqdm(self.data.get(split, [])):
                    if getattr(self.p, 'disable_plm', False):
                        text_ids, text_mask, pred_pos = [], [], 0
                    else:
                        text_ids, text_mask = construct_input_text(sub, rel)
                        pred_pos = text_ids.index(self.mask_token_id)
                    self.triples['{}_{}'.format(split, 'tail')].append(
                        {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)], 'text_ids': text_ids, 'text_mask': text_mask, 'pred_pos': pred_pos})


                    rel_inv = rel + self.p.num_rel

                    if getattr(self.p, 'disable_plm', False):
                        text_ids, text_mask, pred_pos = [], [], 0
                    else:
                        text_ids, text_mask = construct_input_text(obj, rel_inv)
                        pred_pos = text_ids.index(self.mask_token_id)
                    self.triples['{}_{}'.format(split, 'head')].append(
                        {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)], 'text_ids': text_ids,
                         'text_mask': text_mask, 'pred_pos': pred_pos})


                print('{}_{} num is {}'.format(split, 'tail', len(self.triples['{}_{}'.format(split, 'tail')])))
                print('{}_{} num is {}'.format(split, 'head', len(self.triples['{}_{}'.format(split, 'head')])))


            self.triples = dict(self.triples)
            json.dump(self.triples, open(triples_save_file, 'w'))
            print(f"[Info] Generated triples: train={len(self.triples.get('train', []))} samples")

        # Validate training data
        train_samples = len(self.triples.get('train', []))
        if train_samples == 0:
            raise ValueError(f"No training data found! Check data files in {data_path}. "
                           f"Loaded data splits: {list(self.data.keys())}, "
                           f"Data counts: {[(k, len(v)) for k, v in self.data.items()]}")

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples.get(split, []), self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                pin_memory=(self.device.type == 'cuda'),
                persistent_workers=(self.device.type == 'cuda' and self.p.num_workers > 0),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.test_batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.test_batch_size),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.test_batch_size),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.test_batch_size),
        }
        print('num_ent {} num_rel {}\n'.format(self.p.num_ent, self.p.num_rel))
        print('train set num is {}\n'.format(len(self.triples['train'])))
        print('{}_{} num is {}\n'.format('test', 'tail', len(self.triples['{}_{}'.format('test', 'tail')])))
        print(
            '{}_{} num is {}\n'.format('valid', 'tail', len(self.triples['{}_{}'.format('valid', 'tail')])))
        self.edge_index, self.edge_type = self.construct_adj()
        
        self.enable_type_aware = getattr(self.p, 'enable_type_aware', False)
        self.type_aware_nonintrusive = getattr(self.p, 'type_aware_nonintrusive', False)
        self.type_soft_constraint = getattr(self.p, 'type_soft_constraint', False)
        effective_mask_mode = getattr(self.p, 'type_mask_mode', 'auto')
        effective_prior_alpha = getattr(self.p, 'type_prior_alpha', 0.3)
        soft_alpha_cap = getattr(self.p, 'type_soft_alpha', None)
        if self.type_soft_constraint:
            effective_mask_mode = 'soft'
            if soft_alpha_cap is not None:
                effective_prior_alpha = min(effective_prior_alpha, soft_alpha_cap)

        self.type_aware_module = TypeAwareModule(
            num_rel=self.p.num_rel,
            num_ent=self.p.num_ent,
            device=self.device,
            enable_type_aware=self.enable_type_aware,
            prior_mode=getattr(self.p, 'type_prior_mode', 'add'),
            prior_alpha=effective_prior_alpha,
            prior_tau=getattr(self.p, 'type_prior_tau', 1.0),
            prior_eps=getattr(self.p, 'type_prior_eps', 1e-6),
            prior_zero_mean=getattr(self.p, 'type_prior_zero_mean', True),
            mask_mode=effective_mask_mode,
            mask_penalty=getattr(self.p, 'type_mask_penalty', 1.0),
        )
        
        if self.enable_type_aware:
            self.type_aware_module.build_relation_candidate_mask(self.sr2o)
            print(f"[Type-Aware] Relation candidate mask built with shape: {self.type_aware_module.rel_candidate_mask.shape}")
            if self.type_aware_module.rel_candidate_density is not None:
                dens = self.type_aware_module.rel_candidate_density
                print(f"[Type-Aware] Candidate density: mean={dens.mean().item():.4f}, median={dens.median().item():.4f}, min={dens.min().item():.4f}, max={dens.max().item():.4f}")
            self.type_aware_module.build_relation_prior(self.sr2o)
            if self.type_aware_module.rel_prior_probs is not None:
                print(f"[Type-Aware] Relation prior built with shape: {self.type_aware_module.rel_prior_probs.shape}, mode={self.type_aware_module.prior_mode}, alpha={self.type_aware_module.prior_alpha}, tau={self.type_aware_module.prior_tau}, zero_mean={self.type_aware_module.prior_zero_mean}")
                print(f"[Type-Aware] Mask config: mode={self.type_aware_module.mask_mode}, penalty={self.type_aware_module.mask_penalty}")
                print(f"[Type-Aware] NonIntrusive={self.type_aware_nonintrusive}, SoftConstraint={self.type_soft_constraint}")
            self.type_aware_module.build_relation_prior(self.sr2o)
            if self.type_aware_module.rel_prior_probs is not None:
                print(f"[Type-Aware] Relation prior built with shape: {self.type_aware_module.rel_prior_probs.shape}, mode={self.type_aware_module.prior_mode}, alpha={self.type_aware_module.prior_alpha}, tau={self.type_aware_module.prior_tau}, zero_mean={self.type_aware_module.prior_zero_mean}")
                print(f"[Type-Aware] Mask config: mode={self.type_aware_module.mask_mode}, penalty={self.type_aware_module.mask_penalty}")

    def construct_adj(self):

        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)
        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type

    def __init__(self, params):

        self.p = params

        use_gpu = torch.cuda.is_available() and isinstance(self.p.gpu, int) and self.p.gpu >= 0
        if use_gpu:
            self.device = torch.device(f'cuda:{self.p.gpu}')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        if getattr(self.p, 'disable_plm', False):
            self.mask_token_id = 0
        elif self.p.pretrained_model_name.lower() == 'bert_base' or self.p.pretrained_model_name.lower() == 'bert_large':
            self.mask_token_id = 103
        elif self.p.pretrained_model_name.lower() == 'roberta_base' or self.p.pretrained_model_name.lower() == 'roberta_large':
            self.mask_token_id = 50264
        self.load_data()
        self.model = self.add_model(self.p.model)
        self.optimizer, self.optimizer_mi = self.add_optimizer(self.model)

        self.best_val_mrr = {'combine': 0., 'struc': 0., 'text': 0.}
        self.best_epoch = {'combine': 0, 'struc': 0, 'text': 0}

        os.makedirs('./checkpoints', exist_ok=True)

        if self.p.load_path != None and self.p.load_epoch > 0 and self.p.load_type != '':
            self.path_template = os.path.join('./checkpoints', self.p.load_path)
            path = self.path_template + '_type_{0}_epoch_{1}'.format(self.p.load_type, self.p.load_epoch)
            self.load_model(path)
            print('Successfully Loaded previous model')
        else:
            print('Training from Scratch ...')
            self.path_template = os.path.join('./checkpoints', self.p.name)


    def add_model(self, model):
        """
        Creates the computational graph

        Parameters
        ----------
        model_name:     Contains the model name to be created

        Returns
        -------
        Creates the computational graph for model and initializes it

        """

        model_name = model

        if model_name.lower() == 'disenkgat':
            model = DisenCSPROM(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'adaptive_multihop':
            from adaptive_multihop import AdaptiveMultiHopPDKGC
            model = AdaptiveMultiHopPDKGC(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'pretrained_disenkgat':
            if self.p.dataset =='FB15k-237':
                model_save_path = '/home/zjlab/gengyx/KGE/DisenKGAT-2023/checkpoints/ConvE_FB15k_K4_D200_club_b_mi_drop_200d_08_09_2023_19:21:24'
            elif self.p.dataset == 'WN18RR':
                model_save_path = '/home/zjlab/gengyx/KGE/DisenKGAT-2023/checkpoints/ConvE_wn18rr_K2_D200_club_b_mi_drop_200d_27_09_2023_17:12:54'
            state = torch.load(model_save_path, map_location=self.device)
            pretrained_dict = state['state_dict']
            model = DisenCSPROM(self.edge_index, self.edge_type, params=self.p)
            model_dict = model.state_dict()
            pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            raise NotImplementedError

        model.to(self.device)
        return model

    def add_optimizer(self, model):


        mi_disc_params = list(map(id, model.mi_Discs.parameters()))
        rest_params = [p for p in model.parameters() if id(p) not in mi_disc_params and p.requires_grad]
        if len(rest_params) == 0:
            rest_params = [p for p in model.parameters() if id(p) not in mi_disc_params]
        if len(rest_params) == 0:
            rest_params = [p for p in model.parameters() if p.requires_grad]
        if len(rest_params) == 0:
            rest_params = list(model.parameters())
        mi_params = [p for p in model.mi_Discs.parameters() if p.requires_grad]
        if len(mi_params) == 0:
            mi_params = list(model.mi_Discs.parameters())
        named_params = [(n, p) for (n, p) in model.named_parameters() if id(p) not in mi_disc_params]
        no_decay_keywords = ['bias', 'LayerNorm.weight', 'LayerNorm.bias', 'bn.weight', 'bn.bias', 'norm.weight', 'norm.bias']
        decay_group = [p for (n, p) in named_params if p.requires_grad and not any(k in n for k in no_decay_keywords)]
        no_decay_group = [p for (n, p) in named_params if p.requires_grad and any(k in n for k in no_decay_keywords)]
        if len(decay_group) + len(no_decay_group) == 0:
            decay_group = rest_params
            no_decay_group = []
        main_optimizer = torch.optim.Adam([
            {'params': decay_group, 'weight_decay': self.p.l2},
            {'params': no_decay_group, 'weight_decay': 0.0},
        ], lr=self.p.lr)

        mi_named_params = list(model.mi_Discs.named_parameters())
        mi_decay_group = [p for (n, p) in mi_named_params if p.requires_grad and not any(k in n for k in no_decay_keywords)]
        mi_no_decay_group = [p for (n, p) in mi_named_params if p.requires_grad and any(k in n for k in no_decay_keywords)]
        if len(mi_decay_group) + len(mi_no_decay_group) == 0:
            mi_decay_group = mi_params
            mi_no_decay_group = []
        mi_optimizer = torch.optim.Adam([
            {'params': mi_decay_group, 'weight_decay': self.p.l2},
            {'params': mi_no_decay_group, 'weight_decay': 0.0},
        ], lr=self.p.lr)

        return main_optimizer, mi_optimizer



    def read_batch(self, batch, split):
        if split == 'train':
            triple, label, text_ids, text_mask, pred_pos = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label, text_ids, text_mask, pred_pos
        else:
            triple, label, text_ids, text_mask, pred_pos = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label, text_ids, text_mask, pred_pos


    def load_model(self, load_path):
        state = torch.load(load_path, map_location=self.device)
        state_dict = state.get('state_dict', state)
        model_dict = self.model.state_dict()
        compatible_state = {}
        skipped_keys = []
        for k, v in state_dict.items():
            if k in model_dict and isinstance(v, torch.Tensor) and model_dict[k].shape == v.shape:
                compatible_state[k] = v
            else:
                skipped_keys.append(k)
        missing_before = [k for k in model_dict.keys() if k not in compatible_state]
        load_res = self.model.load_state_dict(compatible_state, strict=False)
        if 'best_val_mrr' in state:
            try:
                self.best_val_mrr[self.p.load_type] = state['best_val_mrr']
            except Exception:
                pass
        try:
            self.best_epoch[self.p.load_type] = self.p.load_epoch
        except Exception:
            pass
        try:
            if 'optimizer' in state and hasattr(self, 'optimizer') and self.optimizer is not None:
                self.optimizer.load_state_dict(state['optimizer'])
        except Exception as e:
            print(f"[Warn] Optimizer state not loaded: {e}")
        print(f"[Info] Loaded params: {len(compatible_state)}/{len(model_dict)}; Skipped: {len(skipped_keys)}; Missing in checkpoint: {len(load_res.missing_keys)}; Unexpected ignored: {len(skipped_keys)}")
        if len(skipped_keys) > 0:
            print(f"[Info] Examples of skipped keys: {skipped_keys[:5]}")
        return self.model

    def evaluate(self, split, epoch):

        left_results = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        
        if self.enable_type_aware and len(left_results) == 7:
            left_combine_results, left_struc_results, left_lm_results, left_base_combine, left_base_struc, left_base_lm, left_stats = left_results
            right_combine_results, right_struc_results, right_lm_results, right_base_combine, right_base_struc, right_base_lm, right_stats = right_results
            
            log_combine_res, combine_results = get_and_print_combined_results(left_combine_results, right_combine_results)
            log_struc_res, struc_results = get_and_print_combined_results(left_struc_results, right_struc_results)
            log_lm_res, lm_results = get_and_print_combined_results(left_lm_results, right_lm_results)
            
            log_base_combine_res, base_combine_results = get_and_print_combined_results(left_base_combine, right_base_combine)
            log_base_struc_res, base_struc_results = get_and_print_combined_results(left_base_struc, right_base_struc)
            log_base_lm_res, base_lm_results = get_and_print_combined_results(left_base_lm, right_base_lm)
            
            all_stats = left_stats + right_stats
            coverage_stats = self.type_aware_module.get_coverage_stats(all_stats)
            
            if getattr(self.p, 'type_aware_nonintrusive', False):
                print('[Evaluating Epoch {} {}]: \n COMBINE results (Baseline) {}'.format(epoch, split, log_base_combine_res))
            else:
                print('[Evaluating Epoch {} {}]: \n COMBINE results (Type-Aware) {}'.format(epoch, split, log_combine_res))
            
            if getattr(self.p, 'report_type_aware_effect', True):
                baseline_results = {
                    'combine': base_combine_results,
                    'structure': base_struc_results,
                    'lm': base_lm_results
                }
                self.type_aware_module.print_type_aware_report(coverage_stats, 
                                                              {'combine': combine_results, 'structure': struc_results, 'lm': lm_results},
                                                              baseline_results)
        else:
            left_combine_results, left_struc_results, left_lm_results = left_results
            right_combine_results, right_struc_results, right_lm_results = right_results
            
            log_combine_res, combine_results = get_and_print_combined_results(left_combine_results, right_combine_results)
            log_struc_res, struc_results = get_and_print_combined_results(left_struc_results, right_struc_results)
            log_lm_res, lm_results = get_and_print_combined_results(left_lm_results, right_lm_results)
            
            print('[Evaluating Epoch {} {}]: \n COMBINE results {}'.format(epoch, split, log_combine_res))

        return combine_results, struc_results, lm_results

    def predict(self, split='valid', mode='tail_batch'):

        self.model.eval()

        with torch.no_grad():
            combine_results = {}
            lm_results = {}
            struc_results = {}
            base_combine_results = {}
            base_lm_results = {}
            base_struc_results = {}
            type_aware_stats_list = []
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label, text_ids, text_mask, pred_pos = self.read_batch(batch, split)
                if getattr(self.p, 'disable_plm', False):
                    text_ids, text_mask, pred_pos = None, None, None
                pred, output, _ = self.model.forward(sub, rel, text_ids, text_mask, pred_pos, split)
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_output = output[b_range, obj]
                target_pred = pred[b_range, obj]
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 1e9, pred)
                output = torch.where(label.byte(), -torch.ones_like(output) * 1e9, output)
                output[b_range, obj] = target_output
                pred[b_range, obj] = target_pred

                struc_ranks_base = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
                lm_ranks_base = 1 + torch.argsort(torch.argsort(output, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
                
                struc_ranks_base = struc_ranks_base.float()
                lm_ranks_base = lm_ranks_base.float()
                combine_ranks_base = torch.min(torch.stack([struc_ranks_base, lm_ranks_base], dim=1), dim=1)[0]
                combine_ranks_base = combine_ranks_base.float()
                
                base_combine_results['count'] = torch.numel(combine_ranks_base) + base_combine_results.get('count', 0.0)
                base_combine_results['mr'] = torch.sum(combine_ranks_base).item() + base_combine_results.get('mr', 0.0)
                base_combine_results['mrr'] = torch.sum(1.0 / combine_ranks_base).item() + base_combine_results.get('mrr', 0.0)
                for k in range(10):
                    base_combine_results['hits@{}'.format(k + 1)] = torch.numel(combine_ranks_base[combine_ranks_base <= (k + 1)]) + base_combine_results.get('hits@{}'.format(k + 1), 0.0)
                
                base_struc_results['count'] = torch.numel(struc_ranks_base) + base_struc_results.get('count', 0.0)
                base_struc_results['mr'] = torch.sum(struc_ranks_base).item() + base_struc_results.get('mr', 0.0)
                base_struc_results['mrr'] = torch.sum(1.0 / struc_ranks_base).item() + base_struc_results.get('mrr', 0.0)
                for k in range(10):
                    base_struc_results['hits@{}'.format(k + 1)] = torch.numel(struc_ranks_base[struc_ranks_base <= (k + 1)]) + base_struc_results.get('hits@{}'.format(k + 1), 0.0)
                
                base_lm_results['count'] = torch.numel(lm_ranks_base) + base_lm_results.get('count', 0.0)
                base_lm_results['mr'] = torch.sum(lm_ranks_base).item() + base_lm_results.get('mr', 0.0)
                base_lm_results['mrr'] = torch.sum(1.0 / lm_ranks_base).item() + base_lm_results.get('mrr', 0.0)
                for k in range(10):
                    base_lm_results['hits@{}'.format(k + 1)] = torch.numel(lm_ranks_base[lm_ranks_base <= (k + 1)]) + base_lm_results.get('hits@{}'.format(k + 1), 0.0)

                scores_pred, scores_output, type_aware_stats = self.type_aware_module.apply_type_aware(
                    pred, output, rel, obj
                )
                type_aware_stats_list.append(type_aware_stats)

                struc_ranks = 1 + torch.argsort(torch.argsort(scores_pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
                lm_ranks = 1 + torch.argsort(torch.argsort(scores_output, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

                struc_ranks1 = torch.unsqueeze(struc_ranks, 1)
                lm_ranks1 = torch.unsqueeze(lm_ranks, 1)
                combine_ranks = torch.cat((struc_ranks1, lm_ranks1), 1)
                combine_ranks = torch.squeeze(torch.min(combine_ranks, 1)[0])

                combine_ranks = combine_ranks.float()
                struc_ranks = struc_ranks.float()
                lm_ranks = lm_ranks.float()

                combine_results['count'] = torch.numel(combine_ranks) + combine_results.get('count', 0.0)
                combine_results['mr'] = torch.sum(combine_ranks).item() + combine_results.get('mr', 0.0)
                combine_results['mrr'] = torch.sum(1.0 / combine_ranks).item() + combine_results.get('mrr', 0.0)
                for k in range(10):
                    combine_results['hits@{}'.format(k + 1)] = torch.numel(combine_ranks[combine_ranks <= (k + 1)]) + combine_results.get(
                        'hits@{}'.format(k + 1), 0.0)

                struc_results['count'] = torch.numel(struc_ranks) + struc_results.get('count', 0.0)
                struc_results['mr'] = torch.sum(struc_ranks).item() + struc_results.get('mr', 0.0)
                struc_results['mrr'] = torch.sum(1.0 / struc_ranks).item() + struc_results.get('mrr', 0.0)
                for k in range(10):
                    struc_results['hits@{}'.format(k + 1)] = torch.numel(
                        struc_ranks[struc_ranks <= (k + 1)]) + struc_results.get(
                        'hits@{}'.format(k + 1), 0.0)

                lm_results['count'] = torch.numel(lm_ranks) + lm_results.get('count', 0.0)
                lm_results['mr'] = torch.sum(lm_ranks).item() + lm_results.get('mr', 0.0)
                lm_results['mrr'] = torch.sum(1.0 / lm_ranks).item() + lm_results.get('mrr', 0.0)
                for k in range(10):
                    lm_results['hits@{}'.format(k + 1)] = torch.numel(
                        lm_ranks[lm_ranks <= (k + 1)]) + lm_results.get(
                        'hits@{}'.format(k + 1), 0.0)

            if self.enable_type_aware:
                ret_combine = combine_results
                if getattr(self.p, 'type_aware_nonintrusive', False):
                    ret_combine = base_combine_results
                return ret_combine, struc_results, lm_results, base_combine_results, base_struc_results, base_lm_results, type_aware_stats_list
            else:
                return combine_results, struc_results, lm_results

    def run_epoch(self, epoch):

        try:
            if hasattr(self.p, 'warmup_epochs') and isinstance(self.p.warmup_epochs, int) and self.p.warmup_epochs > 0:
                scale = min(1.0, max(1, epoch) / float(self.p.warmup_epochs))
                for g in self.optimizer.param_groups:
                    g['lr'] = self.p.lr * scale
                for g in getattr(self, 'optimizer_mi', {'param_groups': []}).param_groups:
                    g['lr'] = self.p.lr * scale
        except Exception:
            pass

        self.model.train()
        losses = []
        losses_struc = []
        losses_lm = []
        corr_losses = []

        lld_losses = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            self.model.mi_Discs.eval()
            sub, rel, obj, label, text_ids, text_mask, pred_pos = self.read_batch(batch, 'train')
            if getattr(self.p, 'disable_plm', False):
                text_ids, text_mask, pred_pos = None, None, None
            pred, output, corr = self.model.forward(sub, rel, text_ids, text_mask, pred_pos, 'train')

            loss_struc = self.model.loss_fn(pred, label)
            loss_lm = self.model.loss_fn(output, label)
            losses_struc.append(loss_struc.item())
            losses_lm.append(loss_lm.item())

            if self.p.loss_weight:
                loss_weighted_sum = self.model.loss_weight(loss_struc, loss_lm)
                loss = loss_weighted_sum + self.p.alpha * corr
            else:
                loss = loss_struc + loss_lm + self.p.alpha * corr

            corr_losses.append(corr.item())

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            for i in range(self.p.mi_epoch):
                self.model.mi_Discs.train()
                lld_loss = self.model.lld_best(sub, rel)
                self.optimizer_mi.zero_grad()
                lld_loss.backward()
                self.optimizer_mi.step()
                lld_losses.append(lld_loss.item())

            if step % 1000 == 0:
                print(
                    '[E:{}| {}]: Total Loss:{:.5}, Train {} Loss:{:.5}, Train MASK Loss:{:.5}, Corr Loss:{:.5}\t{} \n \t Best Combine Valid MRR: {:.5}'.format(
                        epoch, step, np.mean(losses), self.p.score_func, np.mean(losses_struc), np.mean(losses_lm), np.mean(corr_losses), self.p.name, self.best_val_mrr['combine']))
        loss = np.mean(losses_struc)
        loss_corr = np.mean(corr_losses)
        if self.p.mi_method.startswith('club') and self.p.mi_epoch == 1:
            loss_lld = np.mean(lld_losses)
            return loss, loss_corr, loss_lld
        return loss, loss_corr, 0.

    def save_model(self, results, type, epoch):
        if results['mrr'] > self.best_val_mrr[type]:
            last_path = self.path_template + '_type_{0}_epoch_{1}'.format(type, self.best_epoch[type])
            save_path = self.path_template + '_type_{0}_epoch_{1}'.format(type, epoch)

            self.best_epoch[type] = epoch
            self.best_val_mrr[type] = results['mrr']

            if os.path.exists(last_path) and last_path != save_path:
                os.remove(last_path)
                print('Last Saved Model {} deleted'.format(last_path))
            torch.save({
                'state_dict': self.model.state_dict(),
                'best_val_mrr': self.best_val_mrr[type],
                'optimizer': self.optimizer.state_dict(),
            }, save_path)

    def fit(self):

        try:
            if not self.p.test:

                kill_cnt = 0
                for epoch in range(self.p.load_epoch+1, self.p.max_epochs+1):
                    train_loss, corr_loss, lld_loss = self.run_epoch(epoch)
                    combine_val_results, struc_val_results, lm_val_results = self.evaluate('valid', epoch)
                    if combine_val_results['mrr'] <= self.best_val_mrr['combine']:
                        kill_cnt += 1
                        if kill_cnt % 10 == 0 and self.p.gamma > self.p.max_gamma:
                            self.p.gamma -= 5
                            print('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                        if kill_cnt > self.p.early_stop:
                            print("Early Stopping!!")
                            break
                    else:
                        kill_cnt = 0
                    self.save_model(combine_val_results, 'combine', epoch)
                    self.save_model(struc_val_results, 'struc', epoch)
                    self.save_model(lm_val_results, 'text', epoch)
                    if self.p.mi_method == 'club_s' or self.p.mi_method == 'club_b':
                        print(
                            '[Epoch {}]: Training Loss: {:.5}, corr Loss: {:.5}, lld loss :{:.5}, \n \t Best Combine Valid MRR: {:.5}\n\n'.format(
                                epoch, train_loss, corr_loss,
                                lld_loss, self.best_val_mrr['combine']))
                    else:
                        print(
                            '[Epoch {}]: Training Loss: {:.5}, corr Loss: {:.5}, \n\t Best Combine Valid MRR: {:.5}\n\n'.format(
                                epoch, train_loss, corr_loss,
                                self.best_val_mrr['combine']))

            else:
                print('Loading best model, Evaluating on Test data')
                self.evaluate('test', self.best_epoch[self.p.load_type])
        except Exception as e:
            print ("%s____%s\n"
                              "traceback.format_exc():____%s" % (Exception, e, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name', default='testrun', help='Set run name for saving/restoring models')
    parser.add_argument('-data', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model', dest='model', default='disenkgat', help='Model Name')
    parser.add_argument('-score_func', dest='score_func', default='conve', help='Score Function for Link prediction')
    parser.add_argument('-opn', dest='opn', default='cross', help='Composition Operation to be used in RAGAT')
    parser.add_argument('-loss_weight', dest='loss_weight', action='store_true', help='whether to use weighted loss')
    parser.add_argument('-batch', dest='batch_size', default=2048, type=int, help='Batch size')
    parser.add_argument('-test_batch', dest='test_batch_size', default=2048, type=int,
                        help='Batch size of valid and test data')
    parser.add_argument('-gamma', type=float, default=9.0, help='Margin')
    parser.add_argument('-epoch', dest='max_epochs', type=int, default=1500, help='Number of epochs')
    parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('-num_workers', type=int, default=10, help='Number of processes to construct batches')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')

    parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')

    parser.add_argument('-embed_dim', dest='embed_dim', default=156, type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer', dest='gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop', dest='dropout', default=0.4, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')

    parser.add_argument('-hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('-k_w', dest='k_w', default=12, type=int, help='ConvE: k_w')
    parser.add_argument('-k_h', dest='k_h', default=13, type=int, help='ConvE: k_h')
    parser.add_argument('-num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    parser.add_argument('-logdir', dest='log_dir', default='./log/', help='Log directory')
    parser.add_argument('-config', dest='config_dir', default='./config/', help='Config directory')
    parser.add_argument('-num_bases', dest='num_bases', default=-1, type=int,
                        help='Number of basis relation vectors to use')
    parser.add_argument('-head_num', dest="head_num", default=1, type=int, help="Number of attention heads")
    parser.add_argument('-num_factors', dest="num_factors", default=4, type=int, help="Number of factors")
    parser.add_argument('-alpha', dest="alpha", default=1e-1, type=float, help='Dropout for Feature')
    parser.add_argument('-early_stop', dest="early_stop", default=200, type=int, help="number of early_stop")
    parser.add_argument('-no_act', dest='no_act', action='store_true', help='whether to use non_linear function')
    parser.add_argument('-mi_method', dest='mi_method', default='club_b',
                        help='Composition Operation to be used in RAGAT')
    parser.add_argument('-att_mode', dest='att_mode', default='dot_weight',
                        help='Composition Operation to be used in RAGAT')
    parser.add_argument('-mi_epoch', dest="mi_epoch", default=1, type=int, help="Number of MI_Disc training times")
    parser.add_argument('-score_order', dest='score_order', default='after',
                        help='Composition Operation to be used in RAGAT')
    parser.add_argument('-mi_drop', dest='mi_drop', action='store_true', help='whether to use non_linear function')
    parser.add_argument('-max_gamma', type=float, default=5.0, help='Margin')
    parser.add_argument('-fix_gamma', dest='fix_gamma', action='store_true', help='whether to use non_linear function')
    parser.add_argument('-init_gamma', type=float, default=9.0, help='Margin')
    parser.add_argument('-gamma_method', dest='gamma_method', default='norm',
                        help='Composition Operation to be used in RAGAT')
    parser.add_argument('-gpu', type=int, default=6, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')

    parser.add_argument('-pretrained_model', type=str, default='bert_large', help='Preset name (bert_large/roberta_large/...) or local path to model dir')
    parser.add_argument('-pretrained_model_name', type=str, default='bert_large', help='')
    parser.add_argument('-prompt_hidden_dim', default=-1, type=int, help='')
    parser.add_argument('-text_len', default=72, type=int, help='')
    parser.add_argument('-prompt_length', default=10, type=int, help='')
    parser.add_argument('-desc_max_length', default=40, type=int, help='')
    parser.add_argument('-load_epoch', type=int, default=0)
    parser.add_argument('-load_type', type=str, default='')
    parser.add_argument('-load_path', type=str, default=None)
    parser.add_argument('-test', dest='test', action='store_true', help='test the model')
    parser.add_argument('-unfreeze_layer', type=int, default=0)
    parser.add_argument('-warmup_epochs', type=int, default=0, help='Linear warm-up epochs for learning rate')
    
    parser.add_argument('--dynamic_hops', action='store_true', help='Enable confidence-based dynamic hops')
    parser.add_argument('-confidence_threshold', dest='confidence_threshold', type=float, default=0.8, help='Confidence threshold for dynamic hops')
    parser.add_argument('-hop1_threshold', dest='hop1_threshold', type=float, default=0.5, help='Confidence threshold for 1 hop (>= this value means 1 hop)')
    parser.add_argument('-hop2_threshold', dest='hop2_threshold', type=float, default=0.4, help='Confidence threshold for 2 hops (>= this value and < hop1_threshold means 2 hops)')
    parser.add_argument('-hop3_threshold', dest='hop3_threshold', type=float, default=0.3, help='Confidence threshold for 3 hops (>= this value and < hop2_threshold means 3 hops, effective when 3+ hops are allowed)')
    parser.add_argument('-hop4_threshold', dest='hop4_threshold', type=float, default=0.2, help='Confidence threshold for 4 hops (>= this value and < hop3_threshold means 4 hops, effective when 4+ hops are allowed)')
    parser.add_argument('-max_hops', dest='max_hops', type=int, default=3, help='Maximum hop limit for dynamic hops, should not exceed GCN layers, supports up to 5 hops')
    parser.add_argument('--print_conf_stats', action='store_true', help='Print min/max/avg confidence and 1/2/3 hop sample counts and ratios')
    parser.add_argument('--grouped_backward', action='store_true', help='Group by hop and backpropagate in segments to reduce memory peak')
    parser.add_argument('--enable_amp', dest='enable_amp', action='store_true', help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--grad_clip_norm', type=float, default=0.0, help='Gradient clipping threshold (L2 norm), 0 means no clipping')
    parser.add_argument('--confidence_loss_weight', type=float, default=0.0, help='Confidence auxiliary loss weight (0 means disable supervision)')
    parser.add_argument('--confidence_hidden_dim', type=int, default=128, help='Hidden dimension of confidence estimator')
    parser.add_argument('--disable_plm', action='store_true', help='Disable PLM branch, only use structure branch for scoring (for quick testing)')
    
    parser.add_argument('--enable_type_aware', dest='enable_type_aware', action='store_true', help='Enable type-aware evaluation')
    parser.add_argument('--report_type_aware_effect', dest='report_type_aware_effect', action='store_true', help='Report type-aware effect comparison')
    parser.add_argument('--type_aware_nonintrusive', action='store_true', help='Do not change main evaluation/save: type-aware only for reporting (main results remain baseline)')
    parser.add_argument('--type_soft_constraint', action='store_true', help='Apply type-aware in soft constraint mode (force soft mask, no hard clipping)')
    parser.add_argument('--type_soft_alpha', type=float, default=None, help='Prior strength upper limit under soft constraint (if specified, apply min clipping to type_prior_alpha)')
    parser.add_argument('--type_prior_mode', type=str, default='add', choices=['add', 'mul'], help='Type prior injection mode: add for additive bias, mul for log-domain multiplication')
    parser.add_argument('--type_prior_alpha', type=float, default=0.3, help='Type prior strength coefficient')
    parser.add_argument('--type_prior_tau', type=float, default=1.0, help='Type prior temperature scaling')
    parser.add_argument('--type_prior_eps', type=float, default=1e-6, help='Laplace smoothing term')
    parser.add_argument('--type_prior_zero_mean', action='store_true', help='Zero-mean the prior bias for each relation to avoid overall translation affecting thresholds')
    parser.add_argument('--type_mask_mode', type=str, default='soft', choices=['soft', 'none'], help='Type mask strategy: soft for soft penalty, none for disabled')
    parser.add_argument('--type_mask_penalty', type=float, default=1.0, help='Soft mask penalty magnitude for illegal entities')
    parser.set_defaults(print_conf_stats=False, grouped_backward=False, report_type_aware_effect=False, type_prior_zero_mean=False)
    args = parser.parse_args()

    if args.load_path == None and args.load_epoch == 0 and args.load_type == '':
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H-%M-%S')

    args.pretrained_model_name = args.pretrained_model

    orig_pretrained_model_name = args.pretrained_model_name
    if args.pretrained_model in ['bert_large', 'bert_base', 'roberta_large', 'roberta_base']:
        preset = args.pretrained_model
        base_dir = os.path.dirname(__file__)
        if preset == 'bert_large':
            # Try local paths first, then fallback to server path
            local_dir_primary = os.path.join(base_dir, 'bert_base')  # Use bert_base for bert_large if available
            local_dir_secondary = os.path.join(base_dir, '..', 'KnowledgeGraphEmbedding-master', 'bert_base')
            server_path = '/home/zjlab/gengyx/LMs/BERT_large'
            if os.path.isdir(local_dir_primary):
                args.pretrained_model = local_dir_primary
            elif os.path.isdir(local_dir_secondary):
                args.pretrained_model = local_dir_secondary
            elif os.path.isdir(server_path):
                args.pretrained_model = server_path
            else:
                args.pretrained_model = local_dir_primary  # Default to local path
        elif preset == 'bert_base':
            # Prioritize local relative paths
            local_dir_primary = os.path.join(base_dir, 'bert_base')
            local_dir_secondary = os.path.join(base_dir, '..', 'KnowledgeGraphEmbedding-master', 'bert_base')
            if os.path.isdir(local_dir_primary):
                args.pretrained_model = local_dir_primary  # Use relative path
                print(f"[Info] Using local bert_base at: {args.pretrained_model}")
            elif os.path.isdir(local_dir_secondary):
                args.pretrained_model = local_dir_secondary
                print(f"[Info] Using bert_base from parent directory: {args.pretrained_model}")
            else:
                args.pretrained_model = local_dir_primary  # Default to local path
                print(f"[Warning] bert_base not found at {local_dir_primary}, will try HuggingFace or fail")
        elif preset == 'roberta_large':
            # Try local paths first, then fallback to server path
            local_dir_primary = os.path.join(base_dir, 'roberta_base')
            local_dir_secondary = os.path.join(base_dir, '..', 'KnowledgeGraphEmbedding-master', 'roberta_base')
            server_path = '/home/zjlab/gengyx/LMs/RoBERTa_large'
            if os.path.isdir(local_dir_primary):
                args.pretrained_model = local_dir_primary  # Use relative path
            elif os.path.isdir(local_dir_secondary):
                args.pretrained_model = local_dir_secondary
            elif os.path.isdir(server_path):
                args.pretrained_model = server_path
            else:
                args.pretrained_model = local_dir_primary  # Default to local path
        elif preset == 'roberta_base':
            # Prioritize local relative paths
            local_dir_primary = os.path.join(base_dir, 'roberta_base')
            local_dir_secondary = os.path.join(base_dir, '..', 'KnowledgeGraphEmbedding-master', 'roberta_base')
            if os.path.isdir(local_dir_primary):
                args.pretrained_model = local_dir_primary  # Use relative path
            elif os.path.isdir(local_dir_secondary):
                args.pretrained_model = local_dir_secondary
            else:
                args.pretrained_model = local_dir_primary  # Default to local path
    
    # Check if local model directory exists
    local_dir_exists = os.path.isdir(args.pretrained_model)
    config_exists = os.path.exists(os.path.join(args.pretrained_model, 'config.json')) if local_dir_exists else False
    
    if not local_dir_exists:
        # Directory doesn't exist
        raise ValueError(f"Model directory not found at: {args.pretrained_model}\n"
                        f"Please ensure the model directory exists. Expected paths:\n"
                        f"  - {os.path.join(os.path.dirname(__file__), 'bert_base')}\n"
                        f"  - {os.path.join(os.path.dirname(__file__), '..', 'KnowledgeGraphEmbedding-master', 'bert_base')}\n"
                        f"Current working directory: {os.getcwd()}\n"
                        f"Script directory: {os.path.dirname(__file__)}")
    
    if not config_exists:
        # Directory exists but config.json is missing - try to load anyway (might be in cache or different structure)
        print(f"[Warning] config.json not found in {args.pretrained_model}, but directory exists. Trying to load model...")
    
    # Try to load config - use local_files_only only if we're sure it's a local path (not HuggingFace ID)
    is_hf_id = args.pretrained_model in ['bert-base-uncased', 'bert-large-uncased', 'roberta-base', 'roberta-large']
    try:
        if is_hf_id:
            # This is a HuggingFace ID, but we'll try with local_files_only=False as fallback
            # (though it should have been caught earlier)
            print(f"[Warning] Using HuggingFace ID {args.pretrained_model}, this may require network connection")
            args.vocab_size = AutoConfig.from_pretrained(args.pretrained_model, local_files_only=False).vocab_size
            args.model_dim = AutoConfig.from_pretrained(args.pretrained_model, local_files_only=False).hidden_size
        else:
            # Local path - try with local_files_only first, then without if it fails
            try:
                args.vocab_size = AutoConfig.from_pretrained(args.pretrained_model, local_files_only=True).vocab_size
                args.model_dim = AutoConfig.from_pretrained(args.pretrained_model, local_files_only=True).hidden_size
                print(f"[Info] Successfully loaded model config from local path: {args.pretrained_model}")
            except Exception as local_error:
                # If local_files_only fails, try without it (might be in HuggingFace cache)
                print(f"[Warning] Failed with local_files_only=True, trying without: {local_error}")
                args.vocab_size = AutoConfig.from_pretrained(args.pretrained_model, local_files_only=False).vocab_size
                args.model_dim = AutoConfig.from_pretrained(args.pretrained_model, local_files_only=False).hidden_size
                print(f"[Info] Successfully loaded model config (may be from cache): {args.pretrained_model}")
    except Exception as e:
        raise ValueError(f"Failed to load model config from: {args.pretrained_model}\n"
                        f"Error: {e}\n"
                        f"Directory exists: {local_dir_exists}, config.json exists: {config_exists}\n"
                        f"Please check that the model directory contains the required files (config.json, pytorch_model.bin, etc.)")
    if args.prompt_hidden_dim == -1:
        args.prompt_hidden_dim = args.embed_dim // 2

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and isinstance(args.gpu, int) and args.gpu >= 0:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        print("Using CPU.")

    if getattr(args, 'dynamic_hops', False) or (hasattr(args, 'model') and str(args.model).lower() == 'adaptive_multihop'):
        try:
            from adaptive_multihop import create_adaptive_multihop_runner
            print('[Info] Using AdaptiveMultiHopRunner to enable adaptive hops and confidence auxiliary loss')
            model = create_adaptive_multihop_runner(args)
        except Exception as e:
            print(f"[Warning] Failed to create adaptive multi-hop runner, falling back to standard Runner: {e}")
            model = Runner(args)
    else:
        model = Runner(args)
    model.fit()
