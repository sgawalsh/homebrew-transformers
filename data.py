import pickle, torch, os, random, re, set_device
from tqdm import tqdm
from set_device import device

def create_training_split(ratio = .8):
    dataset, en_vocab, src_max, fr_vocab, tgt_max = create_dataset('europarl-v7.fr-en.en', 'europarl-v7.fr-en.fr')

    train = dataset[:int(ratio * len(dataset))]
    test = dataset[int(ratio * len(dataset)):]
    dump_file(train, "train.pkl")
    dump_file(test, "test.pkl")
    dump_file({'src_vocab': en_vocab, 'src_max': src_max, 'tgt_vocab': fr_vocab, 'tgt_max': tgt_max}, "vocabs.pkl")

def french_regex(text):
    # 1. Add leading apostrophe space
    text = re.sub(r"(\w)'", r"\1 '", text)

    # 2. Add trailing apostrophe space
    text = re.sub(r"'(\w)", r"' \1", text)

    return re.sub(r'(\w+)([.,?!])', r'\1 \2', text) # separate trailing punctuation from words

def create_dataset(srcFile, tgtFile, maxLen = 64): # maxLen determines maximum sentence length for source and target data
    with open(f'{os.getcwd()}//data//{srcFile}', mode='rt', encoding='utf-8') as f:
        srcLines = f.readlines()
    with open(f'{os.getcwd()}//data//{tgtFile}', mode='rt', encoding='utf-8') as f:
        tgtLines = f.readlines()

    dataset = list(zip(srcLines, tgtLines))

    dataList = []
    srcVocab, tgtVocab, srcMaxLength, tgtMaxLength = dict(), dict(), 0, 0

    for linePair in tqdm(dataset):
        searchString = linePair[0] + " " + linePair[1]
        if re.search(r'([.,?!"])(\S)', searchString) or re.search(r'([.,?!]){2,}', searchString) or re.search(r'([()$%&-])', searchString): # skip unusual punctuation
            continue

        srcLine = re.sub(r'(\w+)([.,?!])', r'\1 \2', linePair[0]).replace('\n','').strip().lower().split() + ['<eos>'] # separate trailing puncuation from words
        # tgtLine = ['<bos>'] + re.sub(r'(\w+)([\'])(\s)', r'\1\2', re.sub(r'(\w+)([.,?!])', r'\1 \2', linePair[1])).replace('\n','').strip().lower().split() + ['<eos>'] # and replace apostrophe with trailing space in french
        tgtLine = ['<bos>'] + french_regex(linePair[1]).replace('\n','').strip().lower().split() + ['<eos>']

        if max(len(srcLine), len(tgtLine)) > maxLen:
            continue

        srcMaxLength = max(srcMaxLength, len(srcLine))
        tgtMaxLength = max(tgtMaxLength, len(tgtLine))
        for word in srcLine:
            srcVocab[word] = None
        for word in tgtLine:
            tgtVocab[word] = None
        
        dataList.append([srcLine, tgtLine])

    return dataList, set_vocab(srcVocab), srcMaxLength, set_vocab(tgtVocab), tgtMaxLength

def set_vocab(vocab):
    sortedKeys = sorted(list(vocab.keys()) + ['<unk>', '<pad>'])
    print(len(sortedKeys))
    for i, word in enumerate(sortedKeys):
        vocab[word] = i
    return vocab

def fix_vocabs():
    with open(f'{os.getcwd()}//data//vocabs.pkl', 'rb') as f:
        vocabs = pickle.load(f)
    src = vocabs['src_vocab']
    tgt = vocabs['tgt_vocab']

    src = set_vocab(src)
    tgt = set_vocab(tgt)

    dump_file({'src_vocab': src, 'src_max': vocabs['src_max'], 'tgt_vocab': tgt, 'tgt_max': vocabs['tgt_max']}, "vocabs.pkl")

def dump_file(obj, fileName):
    with open(f'{os.getcwd()}//data//{fileName}', 'wb+') as f:
        pickle.dump(obj, f)

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
    
    def idx_to_token(self, el):
        return self.inv_data[el]
    
    def to_tokens(self, it):
        output = []

        for el in it:
            output.append(self.to_token(el.item()))
        
        return output

class europarl_data:
    def __init__(self, batch_size = 128):
        
        with open(f'{os.getcwd()}//data//vocabs.pkl', 'rb') as f:
            vocabs = pickle.load(f)

        self.src_vocab = vocab(vocabs['src_vocab'])
        self.tgt_vocab = vocab(vocabs['tgt_vocab'])

        self.batch_size = batch_size
        self.num_steps_src = vocabs['src_max']
        self.num_steps = vocabs['tgt_max']

    def train_dataloader(self, makeNew = False):
        if makeNew or not os.path.exists(f'{os.getcwd()}//data//train_tensors_{device}.pkl'):
            with open(f'{os.getcwd()}//data//train.pkl', 'rb') as f:
                data = pickle.load(f)
                self.trainDataLength = len(data)
                tensors = self.to_tensors(data, self.trainDataLength, self.src_vocab['<pad>'], self.tgt_vocab['<pad>'])
                dump_file(tensors, f"train_tensors_{device}.pkl")
        else:
            with open(f'{os.getcwd()}//data//train_tensors_{device}.pkl', 'rb') as f:
                tensors = pickle.load(f)
            self.trainDataLength = len(tensors[0])
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle = True, generator = torch.Generator(device=set_device.device))
    
    def val_dataloader(self, makeNew = False):
        if makeNew or not os.path.exists(f'{os.getcwd()}//data//test_tensors_{device}.pkl'):
            with open(f'{os.getcwd()}//data//test.pkl', 'rb') as f:
                data = pickle.load(f)
                self.valDataLength = len(data)
                tensors = self.to_tensors(data, self.valDataLength, self.src_vocab['<pad>'], self.tgt_vocab['<pad>'])
                dump_file(tensors, f"test_tensors_{device}.pkl")
        else:
            with open(f'{os.getcwd()}//data//test_tensors_{device}.pkl', 'rb') as f:
                tensors = pickle.load(f)
            self.valDataLength = len(tensors[0])
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=False, generator = torch.Generator(device=set_device.device))
    
    def to_tensors(self, data, dataLength, srcPadInt, tgtPadInt):
        src = torch.full((dataLength, self.num_steps_src), srcPadInt, dtype=torch.int64)
        tgt = torch.full((dataLength, self.num_steps), tgtPadInt, dtype=torch.int64)
        srcLens = torch.empty((dataLength), dtype = torch.int64)
        tgtLens = torch.empty((dataLength), dtype = torch.int64)
        endTarg = torch.full((dataLength, self.num_steps), tgtPadInt, dtype=torch.int64)

        for i, linePair in enumerate(tqdm(data, total=dataLength)):
            for j, word in enumerate(linePair[0]):
                src[i][j] = self.src_vocab[word]
            srcLens[i] = j + 1
            for j, word in enumerate(linePair[1]):
                tgt[i][j] = self.tgt_vocab[word]
            tgtLens[i] = j + 1
            endTarg[i][0 : j] = tgt[i][1 : j + 1] # offset input translations by one for model targets

        return tuple([src, tgt, srcLens, tgtLens, endTarg])
    
    def get_rand_eval(self, n):
        with open(f'{os.getcwd()}//data//test.pkl', 'rb') as f:
            data = pickle.load(f)
        random.shuffle(data)
        return self.to_tensors(data[:n], n, self.src_vocab['<pad>'], self.tgt_vocab['<pad>'])
    
    def build(self, src, targ):
        return self.to_tensors(list(zip([x.split() + ['<eos>'] for x in src], [['<bos>'] + x.split() + ['<eos>'] for x in targ])), self.src_vocab['<pad>'], self.tgt_vocab['<pad>'])