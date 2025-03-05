import os, librosa, pickle, numpy, soundfile, random, torch, data, set_device, nltk

base_channels = 13
phoneme_map = {'<blank>': 0, 'AA' : 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5, 'AY': 6, 'B': 7, 'CH': 8, 'D': 9, 'DH' : 10, 'EH': 11, 'ER': 12, 'EY': 13, 'F': 14, 'G': 15, 'HH': 16, 'IH': 17, 'IY': 18, 'JH': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'NG': 24, 'OW': 25, 'OY': 26, 'P': 27, 'R': 28, 'S': 29, 'SH': 30, 'T': 31, 'TH': 32, 'UH': 33, 'UW': 34, 'V': 35, 'W': 36, 'Y': 37, 'Z': 38, 'ZH': 39, ' ': 40}
letter_map = {'<blank>': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26, "'": 27, " ": 28}

def normalize(x): #change or remove norm?
    return (x - x.mean()) / x.std()

def get_x(x_in):
    return normalize(x_in)

def gen_db_files(relPath = "data\\LibriSpeech\\train-clean-100\\", isTrain = True):
    
    cwd = os.getcwd()
    arpabet = nltk.corpus.cmudict.dict()
    
    for p , d, f in os.walk(f"{cwd}\\{relPath}"):
        if len(f):
            ids = p.split('\\')[-2:]
            with open(os.path.join(p, f"{ids[0]}-{ids[1]}.trans.txt")) as trans:
                for l in [x.split() for x in trans.readlines()]:
                    num = l[0].split('-')[2]
                    words = ["<bos>"] + l[1:] + ["<eos>"]

                    data = soundfile.read(os.path.join(p, f"{ids[0]}-{ids[1]}-{num}.flac"))[0]
                    t_data = librosa.feature.mfcc(y=data, sr=16000, n_mfcc = base_channels, hop_length = 160, win_length = 400) # hop_length = 10 ms, win_length = 25 ms
                    d_1 = numpy.diff(t_data, prepend = 0) # 1st derivative
                    d_2 = numpy.diff(d_1, prepend = 0) # 2nd derivative

                    x = numpy.concatenate([t_data, d_1, d_2]).transpose()
                    
                    with open(os.path.abspath(f"{cwd}\\data\\processed_data\\{"train" if isTrain else "dev"}\\{base_channels}\\{ids[0]}-{ids[1]}-{num}.pkl"), 'wb') as f:
                        pickle.dump([x, words], f)

                    phones = []
                    problem = False
                    
                    for word in words[1:-1]:
                        try:
                            sounds = arpabet[word.lower()][0]
                        except KeyError:
                            word = word.replace("'", "") 
                            try:
                                sounds = arpabet[word.lower()][0]
                            except KeyError:
                                problem = True
                                break
                        
                        sounds = [sound.strip("012") for sound in sounds] # ignore syllable accents
                        phones.append(sounds)
                    
                    if not problem:
                        with open(os.path.abspath(f"{cwd}\\data\\processed_data\\{"train" if isTrain else "dev"}\\{base_channels}\\phonemes\\{ids[0]}-{ids[1]}-{num}.pkl"), 'wb') as f:
                            pickle.dump(phones, f)


def data_split(ratio = .9, dirPath ="data\\processed_data", isTrain = True, shuffle = True):
    fullPath = f"{os.getcwd()}\\{dirPath}\\{"train" if isTrain else "dev"}\\{base_channels}\\"
    fileList = [f for f in os.listdir(fullPath) if os.path.isfile(fullPath + f)]

    if shuffle:
        random.shuffle(fileList)

    for f in fileList[:int(len(fileList) * ratio)]:
        os.replace(fullPath + f, f"{fullPath}train\\{f}")
        try:
            os.replace(f"{fullPath}phonemes\\{f}", f"{fullPath}phonemes\\train\\{f}")
        except FileNotFoundError:
            pass

    for f in fileList[int(len(fileList) * ratio):]:
        os.replace(fullPath + f, f"{fullPath}eval\\{f}")
        try:
            os.replace(f"{fullPath}phonemes\\{f}", f"{fullPath}phonemes\\eval\\{f}")
        except FileNotFoundError:
            pass

def create_vocab(path = "data\\processed_data", isTrain = True):
    vocab, i, src_max, tgt_max = dict(), 0, 1, 0
    fullPath = f"{os.getcwd()}\\{path}\\{"train" if isTrain else "dev"}\\{base_channels}\\"

    for fileName in [f for f in os.listdir(fullPath) if os.path.isfile(fullPath + f)]:

        with open(f'{fullPath}{fileName}', 'rb') as file:
            f = pickle.load(file)

        src_max = max(src_max, f[0].shape[0])
        tgt_max = max(tgt_max, len(f[1]))

        for word in f[1]:
            try:
                vocab[word]
            except KeyError:
                vocab[word] = i
                i += 1

    with open(f'{os.getcwd()}//data//LibriSpeechVocab.pkl', 'wb') as f:
        pickle.dump({"vocab": vocab, "src_max": src_max, "tgt_max": tgt_max}, f)

def create_ctc_vocab(isTrain = True):
    with open(f'{os.getcwd()}//data//LibriSpeechVocab.pkl', 'rb') as f:
        vocabData = pickle.load(f)

    vocab = vocabData["vocab"]

    del vocab["<eos>"]
    del vocab["<bos>"]
    del vocab["<pad>"]

    for i, key in enumerate(vocab.keys(), 1):
        vocab[key] = i

    vocab["<blank>"] = 0

    with open(f'{os.getcwd()}//data//LibriSpeechCtcVocab.pkl', 'wb') as f:
        pickle.dump({"vocab": vocab}, f)

def data_generator_torch_batch(batch_size, isEval, isTrain = True, path = "data\\processed_data\\"):

    path = f"{path}{"eval" if isEval else "train"}\\"
    
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    random.shuffle(files)

    with open(f'{os.getcwd()}//data//LibriSpeechVocab.pkl', 'rb') as f:
        vocabData = pickle.load(f)

    vocab = vocabData["vocab"]
    tgt_max = vocabData["tgt_max"]
            
    for f_batch in batch(files, batch_size):
        x, y, x_lengths, y_targ = [], torch.full((batch_size, tgt_max), vocab["<pad>"], dtype=torch.int32, device=set_device.device), torch.zeros(batch_size, dtype=torch.int32, device=set_device.device), torch.full((batch_size, tgt_max), vocab["<pad>"], dtype=torch.int64, device=set_device.device)
        
        for i, f in enumerate(f_batch):
            xy = pickle.load(open(f"{path}{f}", "rb"))
            
            sentence = [vocab[word] for word in xy[1]]
            x_lengths[i] = xy[0].shape[0]

            x.append(torch.from_numpy(xy[0]).to(set_device.device))
            y[i][:len(sentence)] = torch.Tensor(sentence)
            y_targ[i][:len(sentence) - 1] = y[i][1:len(sentence)]
        
        yield x, y, x_lengths, y_targ

def getFiles(dirPath):
    files = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
    random.shuffle(files)
    return files

def ctc_generator(batch_size, isEval, isTrain = True, mode = "word", path = "data\\processed_data\\"): #TODO phoneme case
    files = getFiles(f"{path}{"phonemes\\" if mode == "phoneme" else ""}{"eval" if isEval else "train"}\\")

    match mode:
        case "word":
            with open(f'{os.getcwd()}//data//LibriSpeechCtcVocab.pkl', 'rb') as f:
                vocab = pickle.load(f)["vocab"]
            def to_sequence(wordList, **kwargs):
                return [vocab[word] for word in wordList]
        case "phoneme":
            vocab = phoneme_map
            phoneDir = f"{path}phonemes\\{"eval" if isEval else "train"}\\"
            def to_sequence(_, fileName):
                with open(f"{phoneDir}{fileName}", 'rb') as f:
                    ret = [vocab[phone] for word in pickle.load(f) for phone in word + [' ']]
                    ret.pop()
                    return ret
        case "letter":
            vocab = letter_map
            def to_sequence(wordList, **kwargs):
                return [vocab[l] for l in " ".join(wordList)]
        case _:
            raise Exception("Invalid mode")
        
    yield len(files)
    path = f"{path}{"eval" if isEval else "train"}\\"
    for f_batch in batch(files, batch_size):
        x, x_lengths, y_targ = [], torch.zeros(batch_size, dtype=torch.int32, device=set_device.device), []
        
        for i, f in enumerate(f_batch):
            xy = pickle.load(open(f"{path}{f}", "rb"))
            
            y_targ.append(to_sequence(xy[1][1:-1], fileName = f))
            x_lengths[i] = xy[0].shape[0]

            x.append(torch.from_numpy(xy[0]).to(set_device.device))
        
        yield x, x_lengths, y_targ

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def gen_files_vocab_split(isTrain = True):
    # gen_db_files(isTrain=isTrain)
    create_vocab(isTrain=isTrain)
    data_split(isTrain=isTrain)

class audio_data:
    def __init__(self, batch_size = 128, dirPath = "data\\processed_data\\", isTrain = True, isCtc = False, mode = "word"):
        
        if isCtc:
            match mode:
                case "word":
                    with open(f'{os.getcwd()}//data//libriSpeechCtcVocab.pkl', 'rb') as f:
                        vocabData = pickle.load(f)
                    self.tgt_vocab = data.vocab(vocabData["vocab"])
                case "phoneme":
                    self.tgt_vocab = data.vocab(phoneme_map)
                case "letter":
                    self.tgt_vocab = data.vocab(letter_map)
                case _:
                    raise Exception("Invalid mode") 
        else:
            with open(f'{os.getcwd()}//data//{"train" if isTrain else "dev"}libriSpeechVocab.pkl', 'rb') as f:
                vocabData = pickle.load(f)

            self.tgt_vocab = data.vocab(vocabData["vocab"])
            self.num_steps_src = vocabData['src_max']
            self.num_steps = vocabData['tgt_max']

        

        self.batch_size = batch_size
        self.dirPath = f"{os.getcwd()}\\{dirPath}{"train" if isTrain else "dev"}\\{base_channels}\\"
    
    def ctc_train_dataloader(self, mode):
        return ctc_generator(self.batch_size, False, True, mode, self.dirPath)
    
    def ctc_val_dataloader(self, mode):
        return ctc_generator(self.batch_size, True, True, mode, self.dirPath)
    
def decodeCtc(sequence, decoder, mode):
    prev = -1
    output = []
    for el in sequence:
        if el == prev:
            continue
        else:
            prev = el
            if el:
                output.append(decoder[el if isinstance(el, int) else el.item()])

    if mode == "word":
        return " ".join(output)
    elif mode in ["letter", "phoneme"]:
        return "".join(output)
    else:
        return output
    
def squashCtc(sequence):
    prev = -1
    output = []
    for el in sequence:
        if el == prev:
            continue
        else:
            prev = el
            if el:
                output.append(el if isinstance(el, int) else el.item())
    return output
    
def decodeTarg(sequence, decoder, mode):
    output = []
    for el in sequence:
        output.append(decoder[el.item()])

    if mode == "word":
        return " ".join(output)
    elif mode == "letter":
        return "".join(output)
    else:
        return output

# create_vocab()
create_ctc_vocab()