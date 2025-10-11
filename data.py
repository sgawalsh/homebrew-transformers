import pickle, torch, os, random, re, tokenizers, math
from tqdm import tqdm
from settings import MAX_LEN, MIN_LEN, SRC_LANG, TRG_LANG, DATA_MODE, dataFileDict

def train_shared_bpe(fileList, vocab_size=32000):
    tokenizer = tokenizers.SentencePieceBPETokenizer()
    tokenizer.train(
        files=fileList,
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )

    return tokenizer

def zip_source_target(srcFile, tgtFile):
    with open(f'{os.getcwd()}//data//{srcFile}', mode='rt', encoding='utf-8') as f:
        srcLines = f.readlines()
    with open(f'{os.getcwd()}//data//{tgtFile}', mode='rt', encoding='utf-8') as f:
        tgtLines = f.readlines()

    return list(zip(srcLines, tgtLines))

def clean_data(srcFile, tgtFile, outSrcFile, outTgtFile):
    dataset = zip_source_target(srcFile, tgtFile)

    with open(f'{os.getcwd()}//data//{outSrcFile}', 'w', encoding='utf-8') as fSrc, open(f'{os.getcwd()}//data//{outTgtFile}', 'w', encoding='utf-8') as fTrg:
        for linePair in tqdm(dataset):
            searchString = linePair[0] + " " + linePair[1]
            if re.search(r'([.,?!"])(\S)', searchString) or re.search(r'([.,?!]){2,}', searchString) or re.search(r'([()$%&-])', searchString): # skip unusual punctuation
                continue

            fSrc.write(re.sub(r'(\w+)([.,?!])', r'\1 \2', linePair[0]).replace('\n','').strip().lower() + '\n') # separate trailing puncuation from words
            fTrg.write(french_regex(linePair[1]).replace('\n','').strip().lower() + '\n')

def create_dataset(srcFile, tgtFile):
    tokenizer = train_shared_bpe([f'{os.getcwd()}//data//{srcFile}', f'{os.getcwd()}//data//{tgtFile}'])
    dataset = zip_source_target(srcFile, tgtFile)

    dataList = encode_dataset(dataset, tokenizer)

    return dataList, tokenizer

def encode_dataset(dataset, tokenizer: tokenizers.Tokenizer):
    dataList = []

    for i, (srcSentence, tgtSentence) in tqdm(enumerate(dataset)):
        srcLine = tokenizer.encode(srcSentence.strip().lower() + '<eos>').ids
        tgtLine = tokenizer.encode('<bos>' + tgtSentence.strip().lower() + '<eos>').ids

        if max(len(srcLine), len(tgtLine)) > MAX_LEN or min(len(srcLine), len(tgtLine)) < MIN_LEN: # skip sentences that are too long or too short
            continue
        
        dataList.append([i, srcLine, tgtLine])

    return dataList

def create_training_split_europarl(evalNum=1500, randomSeed=0, shuffle = True):
    srcFile = dataFileDict[DATA_MODE]["src"]
    tgtFile = dataFileDict[DATA_MODE]["tgt"]
    dataset, src_tgt_shared_tokenizer = create_dataset(srcFile, tgtFile)
    if shuffle:
        if randomSeed is not None:
            random.seed(randomSeed)
        random.shuffle(dataset)
    train = dataset[:-evalNum]
    test = dataset[-evalNum:]
    dump_file(train, f"train_{SRC_LANG}-{TRG_LANG}_bpe.pkl")
    dump_file(test, f"test_{SRC_LANG}-{TRG_LANG}_bpe.pkl")
    src_tgt_shared_tokenizer.save(f"data//{SRC_LANG}_{TRG_LANG}_shared_tokenizer.json")

def create_training_split_wmt():
    srcFile = dataFileDict[DATA_MODE]["src_train"]
    tgtFile = dataFileDict[DATA_MODE]["tgt_train"]
    trainDataset, src_tgt_shared_tokenizer = create_dataset(srcFile, tgtFile)
    testDataset = zip_source_target(dataFileDict[DATA_MODE]["src_test"], dataFileDict[DATA_MODE]["tgt_test"])
    testDataset = encode_dataset(testDataset, src_tgt_shared_tokenizer)

    dump_file(trainDataset, f"train_{SRC_LANG}-{TRG_LANG}_bpe.pkl")
    dump_file(testDataset, f"test_{SRC_LANG}-{TRG_LANG}_bpe.pkl")
    src_tgt_shared_tokenizer.save(f"data//{SRC_LANG}_{TRG_LANG}_shared_tokenizer.json")

def french_regex(text):
    # 1. Add leading apostrophe space
    text = re.sub(r"(\w)'", r"\1 '", text)

    # 2. Add trailing apostrophe space
    text = re.sub(r"'(\w)", r"' \1", text)

    return re.sub(r'(\w+)([.,?!])', r'\1 \2', text) # separate trailing punctuation from words

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
    def __init__(self, maxTokens=2000):
        self.tokenizer = tokenizers.Tokenizer.from_file(f"data//{SRC_LANG}_{TRG_LANG}_shared_tokenizer.json")

        self.max_tokens = maxTokens

    def train_dataloader(self):
        with open(f'{os.getcwd()}//data//train_{SRC_LANG}-{TRG_LANG}_bpe.pkl', 'rb') as f:
            data = pickle.load(f)

        return self.build_dataloader(data)

    def val_dataloader(self):
        with open(f'{os.getcwd()}//data//test_{SRC_LANG}-{TRG_LANG}_bpe.pkl', 'rb') as f:
            data = pickle.load(f)

        return self.build_dataloader(data, False)
    
    def build_dataloader(self, data, shuffleData = True):
        dataset = TranslationDataset(data)
        sampler = BucketedBatchSampler(dataset, max_tokens=self.max_tokens, bucket_size=1000, shuffle=shuffleData)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=lambda batch: collate_fn(batch, pad_id=self.tokenizer.token_to_id('<pad>')),
        )
        return loader
    
    def get_rand_sample(self, n, isEval=True):
        with open(f'{os.getcwd()}//data//{"test" if isEval else "train"}_{SRC_LANG}-{TRG_LANG}_bpe.pkl', 'rb') as f:
            data = pickle.load(f)
        return collate_fn(random.sample(data, n), pad_id=self.tokenizer.token_to_id('<pad>'))
    
# clean_data("europarl-v7.fr-en.en", "europarl-v7.fr-en.fr", "europarl-v7.fr-en.clean.en", "europarl-v7.fr-en.clean.fr")
# create_training_split_europarl()