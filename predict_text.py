import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
#############
# model_checkpoint = os.path.join(os.path.expanduser("~"),"checkpoints/erc/download")
# model_checkpoint = "/home/bdaniela/projects/erc/results/MELD/roberta-large/final/2021-05-17-18-24-48-speaker_mode-None-num_past_utterances-1000-num_future_utterances-1000-batch_size-4-seed-42/hp.json"
# NUM_CLASSES = 7
# NUM_CLASSES = get_num_classes(DATASET)
# model_checkpoint = "/home/bdaniela/projects/erc/results/MELD/roberta-large/final/2021-05-10-09-49-06-speaker_mode-upper-num_past_utterances-0-num_future_utterances-1000-batch_size-4-seed-3/config.json"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# tokenizer = AutoTokenizer.from_pretrained("emoberta-large", use_fast=True)

#############
ROOT_DIR = "./multimodal-datasets/"
NUM_CLASSES = 7
print("upload tokenizer")
tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-large")
print("upload model")
model = AutoModelForSequenceClassification.from_pretrained(
    "tae898/emoberta-large", num_labels=NUM_CLASSES
)
print("finished to upload ")
text = "she is shocked"
input_ids_attention_mask = tokenizer(text)
input_ids = input_ids_attention_mask["input_ids"]
attention_mask = input_ids_attention_mask["attention_mask"]
input_ids = torch.tensor(input_ids).view(-1, len(input_ids))
attention_mask = torch.tensor(attention_mask).view(-1, len(attention_mask))
print("run inference")
outputs = model(
    **{"input_ids": input_ids, "attention_mask": attention_mask},
    # labels=labelid,
    output_attentions=True,
    output_hidden_states=True,
)
attentions = outputs.attentions
pred = int(outputs.logits.argmax().numpy())
id2pred = ["neutral", "joy", "surprise", "anger", "sadness", "disgust","fear"]
print(f"pred = {id2pred[pred]}")
prob_vec = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(f"prob_vec = {prob_vec}")
print("finish")
# pprint.pprint(f"data_idx: {idx}")
# pprint.pprint(f"pred: {ds_test.id2emotion[pred]}")
id2pred = ["neutral", "joy", "surprise", "anger", "sadness", "disgust","fear"]
# id2emotion[id2pred[0]]
# id2emotion = {
#     "neutral": 0.0049800905,
#     "joy": 0.96399665,
#     "surprise": 0.018937444,
#     "anger": 0.0071516023,
#     "sadness": 0.002021492,
#     "disgust": 0.001495996,
#     "fear": 0.0014167271
# }

input_ids, attention_mask, labelid = (
    input_ids,
    attention_mask,
    # random_sample["label"],
)
# labelid = torch.tensor(labelid).view(-1, 1)

#############
# idx, input_ids, attention_mask, labelid, decoded = get_random_sample(
#     ds_test, tokenizer, max_tokens=512
# )
# pprint.pprint(f"{decoded}")
# print()
# outputs = model(**{"input_ids": input_ids, "attention_mask": attention_mask})
outputs = model(
    **{"input_ids": input_ids, "attention_mask": attention_mask},
    # labels=labelid,
    output_attentions=True,
    output_hidden_states=True,
)
attentions = outputs.attentions
pred = int(outputs.logits.argmax().numpy())
truth = int(labelid[0][0].numpy())

# pprint.pprint(f"data_idx: {idx}")
pprint.pprint(f"pred: {ds_test.id2emotion[pred]}")
# pprint.pprint(f"truth: {ds_test.id2emotion[truth]}")
# pprint.pprint(f"number of tokens in the input: {input_ids.shape[1]}")
print()