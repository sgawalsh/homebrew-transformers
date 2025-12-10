import model, torch, data, sacrebleu
from tqdm import tqdm
from tokenizers import Tokenizer

def greedy_eval(n: int, myModel: model.EncoderDecoder, myData: data.source_target_dataloader):
    myModel.eval()
    input = myData.get_rand_sample(n)
    preds, _ = myModel.my_predict_step(input[0], input[2], myData.tokenizer.token_to_id('<bos>'), round(input[0].shape[1] * 1.5))

    print_results(input[0].tolist(), input[-1].tolist(), preds.tolist(), myData.tokenizer)
    print_sacre_bleu(input[-1].tolist(), preds.tolist(), myData.tokenizer)

def beam_eval(myModel: model.EncoderDecoder, n: int):
    myModel.eval()
    myData = data.source_target_dataloader()
    sampleData = list(zip(*myData.get_rand_sample(n)))
    srcList, refList, canList = [], [], []
    for el in tqdm(sampleData):
        pred = myModel.my_beam_search_predict_step(torch.unsqueeze(el[0], 0), torch.unsqueeze(el[2], 0), myData.tokenizer.token_to_id('<bos>'), round(el[0].shape[0] * 1.5))
        srcList.append(torch.squeeze(el[0]).tolist())
        refList.append(torch.squeeze(el[-1]).tolist())
        canList.append(torch.squeeze(pred).tolist())

    print_results(srcList, refList, canList, myData.tokenizer)
    print_sacre_bleu(refList, canList, myData.tokenizer)



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