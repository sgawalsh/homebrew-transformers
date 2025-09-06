import audio_data, data, wav_2_letter, torch, os, settings, ast, random, model, pickle
from torch.nn import functional as F

delim = "*~*"
torch.set_default_device(settings.device)

def prepData():
    mode = "letter"
    myModel = wav_2_letter.wav2letter()
    myModel.load_state_dict(torch.load(f'{os.getcwd()}//models//wav2letter'))

    myData = audio_data.audio_data(batch_size=1, isCtc = True, mode = mode)

    dataGen = myData.ctc_train_dataloader(mode)
    _ = next(dataGen)

    vocab = data.vocab(audio_data.letter_map)

    writeSet(myModel, vocab, myData.ctc_train_dataloader(mode), f'{os.getcwd()}//data//lang_data_train.txt')
    writeSet(myModel, vocab, myData.ctc_val_dataloader(mode), f'{os.getcwd()}//data//lang_data_eval.txt')

def prepVocab():
    vocab, i = {}, 0
    with open(f'{os.getcwd()}//data//lang_data_train.txt', 'r') as f:
        for line in f:
            for word in ast.literal_eval(line.split(delim)[1]):
                try:
                    vocab[word]
                except KeyError:
                    vocab[word] = i
                    i += 1

    with open(f'{os.getcwd()}//data//lang_data_eval.txt', 'r') as f:
        for line in f:
            for word in ast.literal_eval(line.split(delim)[1]):
                try:
                    vocab[word]
                except KeyError:
                    vocab[word] = i
                    i += 1

    vocab['<eos>'] = i
    vocab['<bos>'] = i + 1
    vocab['<pad>'] = i + 2

    with open(f'{os.getcwd()}//data//langVocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)


def writeSet(myModel, vocab, dataGen, filename):
    _ = next(dataGen)

    for batch in dataGen:
        for input in batch[0]:
            pred = myModel(torch.unsqueeze(input.transpose(1, 0), 0).float())
            predLetters = (vocab.to_tokens(pred[0].transpose(1, 0).argmax(-1)))
            targLetters = "".join([vocab.inv_data[el] for el in batch[-1][0]])
            with open(filename, 'a') as f:
                f.write(str(predLetters) + delim + str(targLetters.split()) + "\n")

def getLangMaxesAlg():

    vocab = data.vocab(audio_data.letter_map)

    inputRawMax, inputDecodedMax, outputMax = getLangMaxes(0, 0, 0, f'{os.getcwd()}//data//lang_data_train.txt', vocab)
    inputRawMax, inputDecodedMax, outputMax = getLangMaxes(inputRawMax, inputDecodedMax, outputMax, f'{os.getcwd()}//data//lang_data_eval.txt', vocab)

    with open(f'{os.getcwd()}//data//lang_maxes.pkl', 'wb') as f:
        pickle.dump({"inputRawMax": inputRawMax, "inputDecodedMax": inputDecodedMax, "outputMax": outputMax}, f)

def getLangMaxes(inputRawMax: int, inputDecodedMax: int, outputMax: int, fileName: str, vocab: data.vocab):
    with open(fileName, 'r') as f:
        for line in f:
            xy = line.split(delim)
            x = ast.literal_eval(xy[0])
            inputRawMax = max(inputRawMax, len(x))
            inputDecodedMax = max(inputDecodedMax, len(audio_data.squashCtc([vocab[el] for el in x])))
            outputMax = max(outputMax, len(ast.literal_eval(xy[1])))

    return inputRawMax, inputDecodedMax, outputMax
    
def langDataGen(isEval: bool):
    with open(f'{os.getcwd()}//data//lang_data_{"eval" if isEval else "train"}.txt', 'r') as f:
        lines = f.readlines()

    random.shuffle(lines)

    with open(f'{os.getcwd()}//data//lang_data_temp.txt', 'w') as f:
        f.writelines(lines)

    with open(f'{os.getcwd()}//data//lang_data_temp.txt', 'r') as f:
        for line in f:
            xy = line.split(delim)
            yield ast.literal_eval(xy[0]), xy[1][:-1]

def prepBatch(x, y, src_vocab: data.vocab, tgt_vocab: data.vocab, maxes):
    x = torch.FloatTensor(audio_data.squashCtc([src_vocab[el] for el in x]))
    y = torch.FloatTensor([tgt_vocab[el] for el in y])

class batchPrepper:
    def __init__(self, src_vocab: data.vocab, tgt_vocab: data.vocab):
        self.src = src_vocab
        self.tgt = tgt_vocab
        self.bos = self.tgt["<bos>"]
        self.eos = self.tgt["<eos>"]

        with open(f'{os.getcwd()}//data//lang_maxes.pkl', 'rb') as f:
            self.maxes = pickle.load(f)

    def prepBatch(self, x, y):
        return torch.unsqueeze(torch.Tensor(audio_data.squashCtc([self.src[el] for el in x])), 0).to(settings.device, torch.int32), torch.unsqueeze(torch.Tensor([self.bos] + [self.tgt[el] for el in ast.literal_eval(y)] + [self.eos]), 0).to(settings.device, torch.int64)

def getLoss(Y_hat, Y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        Y_hat = torch.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = torch.reshape(Y, (-1,))
        return F.cross_entropy(Y_hat, Y, reduction='mean' if averaged else 'none', label_smoothing=0.1)

def trainLangModel():
    from settings import modelDict
    modelName = "Full"
    params = modelDict[modelName]

    num_hiddens = params["num_hiddens"]
    num_blks = params["num_blks"]
    dropout = params["dropout"]
    ffn_num_hiddens = params["ffn_num_hiddens"]
    num_heads = params["num_heads"]

    src_vocab = data.vocab(audio_data.letter_map)
    with open(f'{os.getcwd()}//data//langVocab.pkl', 'rb') as f:
        tgt_vocab = data.vocab(pickle.load(f))

    prepper = batchPrepper(src_vocab, tgt_vocab)
    

    encoder = model.TransformerEncoder(len(src_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout)
    decoder = model.TransformerDecoder(len(tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout)
    myModel = model.Seq2Seq(encoder, decoder, tgt_pad=tgt_vocab['<pad>'])
    optim = torch.optim.Adam(myModel.parameters(), lr = 0.00001)

    epochs = 1

    for e in range(epochs):
        dataGen = langDataGen(False)

        for x, y in dataGen:
            x, y = prepper.prepBatch(x, y)
            pred = myModel(x, y[:, :-1], torch.tensor([x.shape[1]]))
            loss = getLoss(pred, y[:, 1:])

            with torch.no_grad():
                optim.zero_grad()
                loss.backward()
                optim.step()
            
            print(loss.item())

trainLangModel()