import pickle, torch, os, random, re, set_device

def create_training_split(ratio = .8):
    dataset, en_vocab, src_max, fr_vocab, tgt_max = create_dataset('europarl-v7.fr-en.en', 'europarl-v7.fr-en.fr')

    train = dataset[:int(ratio * len(dataset))]
    test = dataset[int(ratio * len(dataset)):]
    dump_file(train, "train.pkl")
    dump_file(test, "test.pkl")
    dump_file({'src_vocab': en_vocab, 'src_max': src_max, 'tgt_vocab': fr_vocab, 'tgt_max': tgt_max}, "vocabs.pkl")

def create_dataset(srcFile, tgtFile, maxLen = 16):
    with open(f'{os.getcwd()}//data//{srcFile}', mode='rt', encoding='utf-8') as f:
        srcLines = f.readlines()
    with open(f'{os.getcwd()}//data//{tgtFile}', mode='rt', encoding='utf-8') as f:
        tgtLines = f.readlines()

    dataset = list(zip(srcLines, tgtLines))

    dataList = []
    srcVocab, tgtVocab, srcMaxLength, tgtMaxLength = dict(), dict(), 0, 0

    for linePair in dataset:
        searchString = linePair[0] + " " + linePair[1]
        if re.search(r'([.,?!"])(\S)', searchString) or re.search(r'([.,?!]){2,}', searchString) or re.search(r'([()$%&-])', searchString): # skip unusual punctuation
            continue

        srcLine = re.sub(r'(\w+)([.,?!])', r'\1 \2', linePair[0]).replace('\n','').strip().lower().split() + ['<eos>'] # seperate puncuation from words
        tgtLine = ['<bos>'] + re.sub(r'(\w+)([\'])(\s)', r'\1\2', re.sub(r'(\w+)([.,?!])', r'\1 \2', linePair[1])).replace('\n','').strip().lower().split() + ['<eos>'] # and replace apostrophe with trailing space in french

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
    sorted = list(vocab.keys()) + ['<unk>', '<pad>']
    sorted.sort()
    for i, word in enumerate(sorted):
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


class europoarl_data:
    def __init__(self, batch_size = 128):
        
        with open(f'{os.getcwd()}//data//vocabs.pkl', 'rb') as f:
            vocabs = pickle.load(f)

        self.src_vocab = vocab(vocabs['src_vocab'])
        self.tgt_vocab = vocab(vocabs['tgt_vocab'])

        self.batch_size = batch_size
        self.num_steps_src = vocabs['src_max']
        self.num_steps = vocabs['tgt_max']

    def train_dataloader(self):
        with open(f'{os.getcwd()}//data//train.pkl', 'rb') as f:
            data = pickle.load(f)
            self.trainDataLength = len(data)
            tensors = self.to_tensors(data, self.trainDataLength, self.src_vocab['<pad>'], self.tgt_vocab['<pad>'])
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle = True, generator = torch.Generator(device=set_device.device))
    
    def val_dataloader(self):
        with open(f'{os.getcwd()}//data//test.pkl', 'rb') as f:
            data = pickle.load(f)
            self.valDataLength = len(data)
            tensors = self.to_tensors(data, self.valDataLength, self.src_vocab['<pad>'], self.tgt_vocab['<pad>'])
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=False, generator = torch.Generator(device=set_device.device))
    
    def to_tensors(self, data, dataLength, srcPadInt, tgtPadInt):
        src = torch.full((dataLength, self.num_steps_src), srcPadInt, dtype=torch.int64)
        tgt = torch.full((dataLength, self.num_steps), tgtPadInt, dtype=torch.int64)
        srcLens = torch.empty((dataLength), dtype = torch.int64)
        endTarg = torch.full((dataLength, self.num_steps), tgtPadInt, dtype=torch.int64)

        for i, linePair in enumerate(data):
            for j, word in enumerate(linePair[0]):
                src[i][j] = self.src_vocab[word]
            srcLens[i] = j + 1
            for j, word in enumerate(linePair[1]):
                tgt[i][j] = self.tgt_vocab[word]
            endTarg[i][0 : j] = tgt[i][1 : j + 1]

        return tuple([src, tgt, srcLens, endTarg])
    
    def get_rand_eval(self, n):
        with open(f'{os.getcwd()}//data//test.pkl', 'rb') as f:
            data = pickle.load(f)
        random.shuffle(data)
        return self.to_tensors(data[:n], n, self.src_vocab['<pad>'], self.tgt_vocab['<pad>'])
    
    def build(self, src, targ):
        return self.to_tensors(list(zip([x.split() + ['<eos>'] for x in src], [['<bos>'] + x.split() + ['<eos>'] for x in targ])), self.src_vocab['<pad>'], self.tgt_vocab['<pad>'])