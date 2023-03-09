import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from nltk import sent_tokenize

model_name = 'Vamsi/T5_Paraphrase_Paws'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


def get_paraphrased(model, tokenizer, sentence, num_return_sequences=1, num_beams=5):
    inputs = tokenizer([sentence], return_tensors="pt")
    input_ids, attention_masks = inputs["input_ids"].to(
        device), inputs["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=num_return_sequences
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def paraphrase(p):
    paraphrased = []
    s = sent_tokenize(p)
    for i in s:
        sentence = "paraphrase: " + i + " </s>"
        paraphrased.extend(get_paraphrased(model, tokenizer, sentence))

    return ' '.join(paraphrased)


def main(path):
    with open(path, 'r') as x:
        ds = [json.loads(x) for x in list(x)]

    with open("paraphrased.jsonl", 'w+', encoding='utf-8') as f:
        for s in tqdm(ds):
            paraphrased_answers = []
            chatgpt_answers = s['chatgpt_answers']

            for c in chatgpt_answers:
                paraphrased_answers.append(paraphrase(c))

            s['T5_paraphrased'] = paraphrased_answers

            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main("./reddit_eli5.jsonl")
