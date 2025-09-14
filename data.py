import pickle, torch, os, random, re, tokenizers, math
from tqdm import tqdm
from settings import MAX_LEN, MIN_LEN

def train_shared_bpe(srcFile, tgtFile, vocab_size=32000):
    with open(srcFile, encoding="utf-8") as f:
        src_lines = f.readlines()
    with open(tgtFile, encoding="utf-8") as f:
        tgt_lines = f.readlines()

    all_texts = src_lines + tgt_lines

    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    tokenizer.normalizer = tokenizers.normalizers.Lowercase()
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Metaspace()
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<bos>", "<eos>", "<pad>", "<unk>"]
    )
    tokenizer.train_from_iterator(all_texts, trainer)

    return tokenizer

def clean_data(srcFile, tgtFile, outSrcFile, outTgtFile):
    dataset = zip_source_target(srcFile, tgtFile)

    srcClean, tgtClean = [], []
    for linePair in tqdm(dataset):
        searchString = linePair[0] + " " + linePair[1]
        if re.search(r'([.,?!"])(\S)', searchString) or re.search(r'([.,?!]){2,}', searchString) or re.search(r'([()$%&-])', searchString): # skip unusual punctuation
            continue

        srcClean.append(re.sub(r'(\w+)([.,?!])', r'\1 \2', linePair[0]).replace('\n','').strip().lower() + '<eos>\n') # separate trailing puncuation from words
        # tgtLine = ['<bos>'] + re.sub(r'(\w+)([\'])(\s)', r'\1\2', re.sub(r'(\w+)([.,?!])', r'\1 \2', linePair[1])).replace('\n','').strip().lower().split() + ['<eos>'] # and replace apostrophe with trailing space in french
        tgtClean.append('<bos>'+ french_regex(linePair[1]).replace('\n','').strip().lower() + '<eos>\n')
    
    with open(f'{os.getcwd()}//data//{outSrcFile}', 'w', encoding='utf-8') as f:
        f.writelines(srcClean)

    with open(f'{os.getcwd()}//data//{outTgtFile}', 'w', encoding='utf-8') as f:
        f.writelines(tgtClean)

def create_training_split(ratio = .8):
    dataset, src_tgt_shared_tokenizer = create_dataset('europarl-v7.fr-en.clean.en', 'europarl-v7.fr-en.clean.fr')

    train = dataset[:int(ratio * len(dataset))]
    test = dataset[int(ratio * len(dataset)):]
    dump_file(train, "train_bpe.pkl")
    dump_file(test, "test_bpe.pkl")
    dump_file(src_tgt_shared_tokenizer, "src_tgt_shared_tokenizer.pkl")

def french_regex(text):
    # 1. Add leading apostrophe space
    text = re.sub(r"(\w)'", r"\1 '", text)

    # 2. Add trailing apostrophe space
    text = re.sub(r"'(\w)", r"' \1", text)

    return re.sub(r'(\w+)([.,?!])', r'\1 \2', text) # separate trailing punctuation from words

def zip_source_target(srcFile, tgtFile):
    with open(f'{os.getcwd()}//data//{srcFile}', mode='rt', encoding='utf-8') as f:
        srcLines = f.readlines()
    with open(f'{os.getcwd()}//data//{tgtFile}', mode='rt', encoding='utf-8') as f:
        tgtLines = f.readlines()

    return list(zip(srcLines, tgtLines))

def create_dataset(srcFile, tgtFile): # maxLen determines maximum sentence length for source and target data
    dataset = zip_source_target(srcFile, tgtFile)
    tokenizer = train_shared_bpe(f'{os.getcwd()}//data//{srcFile}', f'{os.getcwd()}//data//{tgtFile}')

    dataList = []

    for srcSentence, tgtSentence in tqdm(dataset):
        srcLine = tokenizer.encode(srcSentence.strip().lower()).ids
        tgtLine = tokenizer.encode(tgtSentence.strip().lower()).ids

        if max(len(srcLine), len(tgtLine)) > MAX_LEN or min(len(srcLine), len(tgtLine)) < MIN_LEN: # skip too long or too short sentences
            continue
        
        dataList.append([srcLine, tgtLine])

    return dataList, tokenizer

def dump_file(obj, fileName):
    with open(f'{os.getcwd()}//data//{fileName}', 'wb+') as f:
        pickle.dump(obj, f)

def collate_fn(batch, pad_id):
    """
    batch: list of (src, tgt) pairs, each a list of token IDs
    """
    src_seqs, tgt_seqs = zip(*batch)

    # Find max lengths in this batch
    src_valid_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    tgt_valid_lens = torch.tensor([len(t) for t in tgt_seqs], dtype=torch.long)
    src_max_len = max(src_valid_lens)
    tgt_max_len = max(tgt_valid_lens)

    # Pad sequences
    src_padded = torch.full((len(batch), src_max_len), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_max_len), pad_id, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_seqs, tgt_seqs)):
        src_padded[i, :len(src)] = torch.tensor(src)
        tgt_padded[i, :len(tgt)] = torch.tensor(tgt)

    # Shift for teacher forcing
    tgt_in  = tgt_padded[:, :-1]  # input to decoder
    tgt_out = tgt_padded[:, 1:]   # expected output

    return src_padded, tgt_in, src_valid_lens, tgt_valid_lens, tgt_out

class BucketedBatchSampler(torch.utils.data.Sampler):
    """
    Groups dataset indices by length, shuffles within buckets, and yields batches
    with roughly `max_tokens` total (src + tgt).
    """

    def __init__(self, dataset, max_tokens=25000, bucket_size=1000, shuffle=True):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.bucket_size = bucket_size
        self.shuffle = shuffle

        # Precompute lengths for sorting/bucketing
        self.lengths = [len(src) + len(tgt) for src, tgt in dataset]

        # Sort dataset indices by length
        self.sorted_indices = sorted(range(len(dataset)), key=lambda i: self.lengths[i])

        # Split sorted indices into buckets
        self.buckets = [
            self.sorted_indices[i:i + bucket_size]
            for i in range(0, len(self.sorted_indices), bucket_size)
        ]

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.buckets)  # shuffle bucket order each epoch

        for bucket in self.buckets:
            if self.shuffle:
                random.shuffle(bucket)  # shuffle inside bucket

            batch, batch_tokens = [], 0
            for idx in bucket:
                n_tokens = self.lengths[idx]
                if batch_tokens + n_tokens > self.max_tokens and batch:
                    yield batch
                    batch, batch_tokens = [], 0

                batch.append(idx)
                batch_tokens += n_tokens

            if batch:
                yield batch

    def __len__(self):
        # Approximate number of batches per epoch
        total_tokens = sum(self.lengths)
        return math.ceil(total_tokens / self.max_tokens)

class vocab:
    def __init__(self, inputDict: dict):
        self.data = inputDict
        self.inv_data = {v: k for k, v in inputDict.items()}

    def __getitem__(self, idx):
        try:
            return self.data[idx]
        except KeyError:
            return self.data["<unk>"]
        
    def __len__(self):
        return len(self.data.keys())
    
    def to_token(self, el):
        return self.inv_data[el]
    
    def to_tokens(self, it):
        output = []

        for el in it:
            output.append(self.to_token(el.item()))
        
        return output

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        # pairs: list of (src_ids, tgt_ids)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]  # returns (src_ids, tgt_ids)

class europarl_data:
    def __init__(self, maxTokens=3000):
        
        with open(f'{os.getcwd()}//data//src_tgt_shared_tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)

        self.max_tokens = maxTokens

    def train_dataloader(self):
        with open(f'{os.getcwd()}//data//train_bpe.pkl', 'rb') as f:
            data = pickle.load(f)

        return self.build_dataloader(data)

    def val_dataloader(self):
        with open(f'{os.getcwd()}//data//test_bpe.pkl', 'rb') as f:
            data = pickle.load(f)

        return self.build_dataloader(data)
    
    def build_dataloader(self, data):
        dataset = TranslationDataset(data)
        sampler = BucketedBatchSampler(dataset, max_tokens=self.max_tokens, bucket_size=1000, shuffle=True)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=lambda batch: collate_fn(batch, pad_id=self.tokenizer.token_to_id('<pad>')),
        )
        return loader
    
    def get_rand_eval(self, n):
        with open(f'{os.getcwd()}//data//test_bpe.pkl', 'rb') as f:
            data = pickle.load(f)
        random.seed(0)
        return collate_fn(random.sample(data, n), pad_id=self.tokenizer.token_to_id('<pad>'))