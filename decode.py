import model, torch, data, trainer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from tokenizers import Tokenizer
from settings import MAX_LEN

def greedy_eval(n: int, myModel: model.EncoderDecoder, myData: data.europarl_data):
    myModel.eval()
    input = myData.get_rand_eval(n)
    preds, _ = myModel.my_predict_step(input[0], input[2], myData.tokenizer.token_to_id('<bos>'), myData.tokenizer.token_to_id('<eos>'), MAX_LEN)

    print_results(input[0].tolist(), input[-1].tolist(), preds.tolist(), myData.tokenizer)
    print_bleu(input[-1].tolist(), preds.tolist(), myData.tokenizer)

def beam_eval(n: int, myModel: model.EncoderDecoder, myData: data.europarl_data):
    myModel.eval()
    inputs = myData.get_rand_eval(n)
    preds = []
    for input in zip(inputs[0], inputs[2]):
        pred = myModel.my_beam_search_predict_step(torch.unsqueeze(input[0], 0), torch.unsqueeze(input[1], 0), myData.tokenizer.token_to_id('<bos>'), myData.tokenizer.token_to_id('<eos>'), MAX_LEN)
        preds.append(torch.squeeze(pred).tolist())

    print_results(inputs[0].tolist(), inputs[-1].tolist(), preds, myData.tokenizer)

def decoder_eval(myModel: model.EncoderDecoder, n: int):
    myModel.eval()
    myData = data.europarl_data()
    eval_dataloader = list(zip(*myData.get_rand_eval(n)))
    srcList, refList, canList = [], [], []
    for el in tqdm(eval_dataloader):
        pred = myModel.my_beam_search_predict_step(torch.unsqueeze(el[0], 0), torch.unsqueeze(el[2], 0), myData.tokenizer.token_to_id('<bos>'), myData.tokenizer.token_to_id('<eos>'), MAX_LEN)
        srcList.append(torch.squeeze(el[0]).tolist())
        refList.append(torch.squeeze(el[-1]).tolist())
        canList.append(torch.squeeze(pred).tolist())

    print_results(srcList, refList, canList, myData.tokenizer)
    print_bleu(refList, canList, myData.tokenizer)

def print_bleu(refs, preds, tokenizer: Tokenizer, smoothing = True):
    bleuWeights = {1: (1, 0, 0, 0), 2: (.5, .5, 0, 0), 3: (.33, .33, .33, 0), 4: (.25, .25, .25, .25)}
    bleuScore = 0.0
    decoded_pred = tokenizer.decode_batch(preds, skip_special_tokens=True)
    decoded_ref  = tokenizer.decode_batch(refs, skip_special_tokens=True)

    for ref, can in zip(decoded_ref, decoded_pred):
        try:
            bleuScore += sentence_bleu([ref], can, weights=bleuWeights[min(4, len(can))], smoothing_function=SmoothingFunction().method1 if smoothing else None)
        except KeyError:
            pass

    print(f'Bleu Score: {bleuScore/len(decoded_pred)}')

def print_results(source, target, predictions, tokenizer: Tokenizer):
    src = tokenizer.decode_batch(source, skip_special_tokens=True)
    tgt = tokenizer.decode_batch(target, skip_special_tokens=True)
    pred = tokenizer.decode_batch(predictions, skip_special_tokens=True)

    for s, t, p in zip(src, tgt, pred):
        print(f'{s} => {t} => {p}')