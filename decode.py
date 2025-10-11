import model, torch, data, sacrebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from tokenizers import Tokenizer
from settings import MAX_LEN

def greedy_eval(n: int, myModel: model.EncoderDecoder, myData: data.europarl_data):
    myModel.eval()
    input = myData.get_rand_sample(n)
    preds, _ = myModel.my_predict_step(input[0], input[2], myData.tokenizer.token_to_id('<bos>'), round(input[0].shape[1] * 1.5))

    print_results(input[0].tolist(), input[-1].tolist(), preds.tolist(), myData.tokenizer)
    print_bleu(input[-1].tolist(), preds.tolist(), myData.tokenizer)

def beam_eval(myModel: model.EncoderDecoder, n: int):
    myModel.eval()
    myData = data.europarl_data()
    sampleData = list(zip(*myData.get_rand_sample(n)))
    srcList, refList, canList = [], [], []
    for el in tqdm(sampleData):
        pred = myModel.my_beam_search_predict_step(torch.unsqueeze(el[0], 0), torch.unsqueeze(el[2], 0), myData.tokenizer.token_to_id('<bos>'), round(el[0].shape[0] * 1.5))
        srcList.append(torch.squeeze(el[0]).tolist())
        refList.append(torch.squeeze(el[-1]).tolist())
        canList.append(torch.squeeze(pred).tolist())

    print_results(srcList, refList, canList, myData.tokenizer)
    print_bleu(refList, canList, myData.tokenizer)
    print_sacre_bleu(refList, canList, myData.tokenizer)

def print_bleu(refs, preds, tokenizer: Tokenizer, smoothing = False):
    bleuWeights = {1: (1, 0, 0, 0), 2: (.5, .5, 0, 0), 3: (.33, .33, .33, 0), 4: (.25, .25, .25, .25)}
    bleuScore = 0.0
    decoded_pred = tokenizer.decode_batch(preds, skip_special_tokens=True)
    decoded_ref  = tokenizer.decode_batch(refs, skip_special_tokens=True)

    for ref, can in zip(decoded_ref, decoded_pred):
        try:
            bleuScore += sentence_bleu([ref.split()], can.split(), weights=bleuWeights[min(4, len(can))], smoothing_function=SmoothingFunction().method1 if smoothing else None)
        except KeyError:
            pass

    print(f'Bleu Score: {bleuScore * 100 / len(decoded_pred):.3f}')


def print_sacre_bleu(refs, preds, tokenizer: Tokenizer, smoothing=False):
    # Decode the batches into strings
    decoded_pred = tokenizer.decode_batch(preds, skip_special_tokens=True)
    decoded_ref  = tokenizer.decode_batch(refs, skip_special_tokens=True)

    # SacreBLEU expects list of hypotheses and list of list-of-references
    # (each reference set can have multiple refs per sentence, but here we just pass one each)
    bleu = sacrebleu.corpus_bleu(
        decoded_pred, 
        [decoded_ref],
        smooth_method='exp' if smoothing else 'none'   # sacrebleu smoothing options
    )

    print(f"sacreBLEU score: {bleu.score:.3f}")

def print_results(source, target, predictions, tokenizer: Tokenizer):
    src = tokenizer.decode_batch(source)
    tgt = tokenizer.decode_batch(target)
    pred = tokenizer.decode_batch(predictions)

    for s, t, p in zip(src, tgt, pred):
        print(f'{s}\n{t}\n{p}\n')