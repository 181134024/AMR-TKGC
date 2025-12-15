from helper import *
from data_loader import *
from model import *
import transformers
from transformers import AutoConfig, BertTokenizer, RobertaTokenizer, AutoModel
transformers.logging.set_verbosity_error()
from tqdm import tqdm


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


class TrainDataset(Dataset):


    def __init__(self, entities, params):
        self.entities = entities
        self.p = params

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, idx):
        ele = self.entities[idx]

        ent = torch.FloatTensor([float(ele['ent_id'])])


        text_ids, text_mask = torch.LongTensor(ele['input_ids']), torch.LongTensor(ele['input_mask'])


        return ent, text_ids, text_mask



    @staticmethod
    def collate_fn(data):
        ent = torch.cat([_[0] for _ in data], dim=0)
        text_ids = pad_sequence([_[1] for _ in data], batch_first=True, padding_value=0)
        text_mask = pad_sequence([_[2] for _ in data], batch_first=True, padding_value=0)

        return ent, text_ids, text_mask




def load_data(p):
    base_dir = os.path.dirname(__file__)

    def read_file2(data_path, filename):
        items = []
        file_name = os.path.join(data_path, filename)
        with open(file_name, encoding='utf-8') as file:
            lines = file.read().strip('\n').split('\n')
        for i in range(1, len(lines)):
            item, id = lines[i].strip().split('\t')
            items.append(item.lower())
        return items

    data_path = os.path.join(base_dir, 'data', p.dataset)
    entities = read_file2(data_path, 'entity2id.txt')

    ent2id = {ent: idx for idx, ent in enumerate(entities)}

    id2ent = {idx: ent for ent, idx in ent2id.items()}


    name_base = os.path.join(base_dir, 'data')
    ent_names = read_file(name_base, p.dataset, 'entityid2name.txt', 'name')
    ent_descs = read_file(name_base, p.dataset, 'entityid2description.txt', 'desc')
    # Use pretrained_model_name if available, otherwise infer from path
    if hasattr(p, 'pretrained_model_name') and p.pretrained_model_name:
        model_name = p.pretrained_model_name.lower()
    else:
        model_name = os.path.basename(p.pretrained_model).split('_')[0].lower()
    
    if 'bert' in model_name:
        tok = BertTokenizer.from_pretrained(p.pretrained_model, add_prefix_space=False)
    elif 'roberta' in model_name:
        tok = RobertaTokenizer.from_pretrained(p.pretrained_model, add_prefix_space=True)
    else:
        # Default to BERT
        tok = BertTokenizer.from_pretrained(p.pretrained_model, add_prefix_space=False)

    triples_save_file = os.path.join(base_dir, 'data', p.dataset, f"{p.pretrained_model_name.lower()}_entity_tokens.txt")

    if os.path.exists(triples_save_file):
        entity_cons = json.load(open(triples_save_file))
    else:
        entity_cons = []
        for ent in id2ent.keys():
            sub_name = ent_names[ent]
            sub_desc = ent_descs[ent]
            sub_text = sub_name + ' ' + sub_desc

            tokenized_text = tok(sub_text, max_length=p.text_len, truncation=True)
            input_ids = tokenized_text.input_ids
            input_mask = tokenized_text.attention_mask
            entity_cons.append({'ent_id': ent, 'input_ids': input_ids, 'input_mask': input_mask})

        json.dump(entity_cons, open(triples_save_file, 'w'))

    return entity_cons, len(ent2id)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-data', dest='dataset', default='WN18RR', help='Dataset to use, default: FB15k-237')

    parser.add_argument('-gpu', type=int, default=6, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')

    parser.add_argument('-pretrained_model', type=str, default='bert_large', choices = ['bert_large', 'bert_base', 'roberta_large', 'roberta_base'])
    parser.add_argument('-pretrained_model_name', type=str, default='bert', help='')
    parser.add_argument('-text_len', default=72, type=int, help='')
    parser.add_argument('-desc_max_length', default=40, type=int, help='')
    parser.add_argument('-num_workers', type=int, default=0, help='Number of processes to construct batches')
    parser.add_argument('-batch', default=20, type=int, help='Batch size')
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available() and isinstance(args.gpu, int) and args.gpu >= 0
    if use_gpu:
        device = torch.device(f'cuda:{args.gpu}')
        try:
            torch.cuda.set_device(args.gpu)
        except Exception:
            pass
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')
    args.pretrained_model_name = os.path.basename(args.pretrained_model).split('_')[0]
    
    orig_pretrained_model_name = args.pretrained_model
    if args.pretrained_model in ['bert_large', 'bert_base', 'roberta_large', 'roberta_base']:
        preset = args.pretrained_model
        if preset == 'bert_large':
            args.pretrained_model = '/home/zjlab/gengyx/LMs/BERT_large'
        elif preset == 'bert_base':
            local_dir_primary = os.path.join(os.path.dirname(__file__), 'bert_base')
            local_dir_secondary = os.path.join(os.path.dirname(__file__), '..', 'KnowledgeGraphEmbedding-master', 'bert_base')
            args.pretrained_model = local_dir_primary if os.path.isdir(local_dir_primary) else local_dir_secondary
        elif preset == 'roberta_large':
            args.pretrained_model = '/home/zjlab/gengyx/LMs/RoBERTa_large'
        elif preset == 'roberta_base':
            local_dir_primary = os.path.join(os.path.dirname(__file__), 'roberta_base')
            local_dir_secondary = os.path.join(os.path.dirname(__file__), '..', 'KnowledgeGraphEmbedding-master', 'roberta_base')
            args.pretrained_model = local_dir_primary if os.path.isdir(local_dir_primary) else local_dir_secondary
    
    try:
        need_fallback = (not os.path.isdir(args.pretrained_model)) or (not os.path.exists(os.path.join(args.pretrained_model, 'config.json')))
        if need_fallback:
            hf_id_map = {
                'bert_large': 'bert-large-uncased',
                'bert_base': 'bert-base-uncased',
                'roberta_large': 'roberta-large',
                'roberta_base': 'roberta-base',
            }
            if orig_pretrained_model_name in hf_id_map:
                args.pretrained_model = hf_id_map[orig_pretrained_model_name]
    except Exception:
        pass

    args.vocab_size = AutoConfig.from_pretrained(args.pretrained_model).vocab_size
    args.model_dim = AutoConfig.from_pretrained(args.pretrained_model).hidden_size
    lm_config = AutoConfig.from_pretrained(args.pretrained_model)
    
    # Set pretrained_model_name for load_data function
    args.pretrained_model_name = orig_pretrained_model_name
    
    lm_model = AutoModel.from_pretrained(args.pretrained_model, config=lm_config).to(device)


    for p in lm_model.parameters():
        p.requires_grad = False

    entity_cons, ent_num = load_data(args)

    data_iter = DataLoader(
        TrainDataset(entity_cons, args),
        batch_size=args.batch,
        shuffle=True,
        num_workers=max(0, args.num_workers),
        collate_fn=TrainDataset.collate_fn
    )

    train_iter = iter(data_iter)


    def read_batch(batch):
        ent, text_ids, text_mask = [_.to(device) for _ in batch]
        return ent, text_ids, text_mask

    text_embeds = torch.zeros([ent_num, args.model_dim], dtype=torch.float)
    for step, batch in tqdm(enumerate(train_iter)):
        ent, text_ids, text_mask = read_batch(batch)
        out = lm_model(input_ids=text_ids, attention_mask=text_mask)

        last_hidden_state = out.last_hidden_state
        sent = torch.mean(last_hidden_state, dim=1)

        for i in range(ent.size(0)):
            ent_id = int(ent[i])
            text_embeds[ent_id] = sent[i].detach().cpu()

    model_basename = os.path.basename(args.pretrained_model).lower()
    if 'bert-base-uncased' in model_basename:
        model_basename = 'bert_base'
    elif 'bert-large-uncased' in model_basename:
        model_basename = 'bert_large'
    elif 'roberta-base' in model_basename:
        model_basename = 'roberta_base'
    elif 'roberta-large' in model_basename:
        model_basename = 'roberta_large'
    embeds_save_file = os.path.join(os.path.dirname(__file__), 'data', args.dataset, f"entity_embeds_{model_basename}.pt")
    torch.save(text_embeds, embeds_save_file)