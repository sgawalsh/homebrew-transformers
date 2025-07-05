import model, torch, data, trainer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

def greedy_eval(n: int, myModel: model.EncoderDecoder, myData: data.europarl_data, myTrainer: trainer.trainer):
    myModel.eval()
    input = myData.get_rand_eval(n)
    preds, _ = myModel.my_predict_step(input[0], input[2], myTrainer.data.tgt_vocab['<bos>'], myData.num_steps)

    print_results(zip(input[0], input[3], preds), myData.src_vocab, myData.tgt_vocab)
    print_bleu(zip([t.tolist() for t in input[3]], [t.tolist() for t in preds]), myTrainer)

def beam_eval(n: int, myModel: model.EncoderDecoder, myData: data.europarl_data, myTrainer: trainer.trainer):
    myModel.eval()
    inputs = myData.get_rand_eval(n)
    preds = []
    for input in zip(inputs[0], inputs[2]):
        pred = myModel.my_beam_search_predict_step(torch.unsqueeze(input[0], 0), torch.unsqueeze(input[1], 0), myTrainer.data.tgt_vocab['<bos>'], myData.num_steps)
        preds.append(torch.squeeze(pred))

    print_results(zip(inputs[0], inputs[3], preds), myData.src_vocab, myData.tgt_vocab)

def decoder_eval(myModel: model.EncoderDecoder, myTrainer: trainer.trainer, n: int):
    myModel.eval()
    myData = data.europarl_data(1)
    # eval_dataloader = myData.val_dataloader()
    eval_dataloader = list(zip(*myData.get_rand_eval(n)))
    srcList, refList, canList = [], [], []
    for el in tqdm(eval_dataloader):
        pred = myModel.my_beam_search_predict_step(torch.unsqueeze(el[0], 0), torch.unsqueeze(el[2], 0), myData.tgt_vocab['<bos>'], myData.num_steps)
        srcList.append(torch.squeeze(el[0]).tolist())
        refList.append(torch.squeeze(el[3]).tolist())
        canList.append(torch.squeeze(pred).tolist())

    print_results(zip([torch.Tensor(l) for l in srcList], [torch.Tensor(l) for l in refList], [torch.Tensor(l) for l in canList]), myData.src_vocab, myData.tgt_vocab)
    print_bleu(zip(refList, canList), myTrainer)

def print_bleu(rc: zip, myTrainer: trainer.trainer, smoothing = True):
    bleuWeights = {1: (1, 0, 0, 0), 2: (.5, .5, 0, 0), 3: (.33, .33, .33, 0), 4: (.25, .25, .25, .25)}
    smoothingFn = SmoothingFunction().method1
    bleuScores, errorCount, bleuCount = 0.0, 0, 0
    for (ref, can) in rc:
        ref, can = myTrainer._trim_eos(ref, can)
        try:
            bleuScores += sentence_bleu([ref], can, weights=bleuWeights[min(4, len(can))], smoothing_function=smoothingFn if smoothing else None)
            bleuCount += 1
        except KeyError:
            errorCount += 1
        
    print(f"BleuScore : {bleuScores /  bleuCount * 100:.3f}")
    if errorCount:
        print(f"Error count: {errorCount}")

def print_results(stp: zip, src_vocab: data.vocab, tgt_vocab: data.vocab):
    for s, t, p in stp:
        src, trans, targ = [], [], []
        for token in src_vocab.to_tokens(s):
            if token == '<eos>':
                break
            src.append(token)
        for token in tgt_vocab.to_tokens(t):
            if token == '<eos>':
                break
            targ.append(token)
        for token in tgt_vocab.to_tokens(p):
            if token == '<eos>':
                break
            trans.append(token)
        print(f'{" ".join(src)} => {" ".join(trans)} => {" ".join(targ)}')