from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from evaluate import load
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
#  You can install and import any other libraries if needed

# Some Chinese punctuations will be tokenized as [UNK], so we replace them with English ones
token_replacement = [
    ["：" , ":"],
    ["，" , ","],
    ["“" , "\""],
    ["”" , "\""],
    ["？" , "?"],
    ["……" , "..."],
    ["！" , "!"]
]

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir="./cache/")

class SemevalDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation", "test"]
        self.data = load_dataset(
            "sem_eval_2014_task_1", split=split, trust_remote_code=True, cache_dir="./cache/"
        ).to_list()

    def __getitem__(self, index):
        d = self.data[index]
        # Replace Chinese punctuations with English ones
        for k in ["premise", "hypothesis"]:
            for tok in token_replacement:
                d[k] = d[k].replace(tok[0], tok[1])
        return d

    def __len__(self):
        return len(self.data)

data_sample = SemevalDataset(split="train").data[:3]
print(f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")

# Define the hyperparameters
# You can modify these values if needed
lr = 5e-5
epochs = 10
train_batch_size = 64
validation_batch_size = 64
test_batch_size = 64
num_labels = 3
regression = 1

# TODO1: Create batched data for DataLoader
# `collate_fn` is a function that defines how the data batch should be packed.
# This function will be called in the DataLoader to pack the data batch.

def collate_fn(batch):
    # TODO1-1: Implement the collate_fn function
    # Write your code here
    # The input parameter is a data batch (tuple), and this function packs it into tensors.
    # Reference: https://www.cnblogs.com/king-lps/p/10990304.html
    premise = [item["premise"] for item in batch]
    hypothesis = [item["hypothesis"] for item in batch]
    relatedness_score = [item["relatedness_score"] for item in batch]
    entailment_judgement = [item["entailment_judgment"] for item in batch]

    # Use tokenizer to pack tokenize and pack the data and its corresponding labels.
    # Reference 1: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__
    # Reference 2: https://github.com/IKMLab/NTHU_Natural_Language_Processing/blob/main/2025/Slides/huggingface_tutorial_bert.pdf
    data_batch = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Reference 1: https://www.geeksforgeeks.org/python/how-to-get-the-data-type-of-a-pytorch-tensor/
    # Reference 2: https://stackoverflow.com/questions/75109379/why-pytorch-dataset-class-does-not-returning-list
    # label_relatedness_score must be a float datatype, like shown in the reference 2 above
    label_relatedness_score = torch.tensor(relatedness_score, dtype=torch.float)
    label_entailment_judgement = torch.tensor(entailment_judgement)
    
    # Return the data batch and labels for each sub-task.
    return data_batch, label_relatedness_score, label_entailment_judgement

# TODO1-2: Define your DataLoader
# Reference: https://discuss.pytorch.org/t/how-to-use-collate-fn/27181/5
dl_train = DataLoader(SemevalDataset(split="train"), shuffle=True, batch_size=train_batch_size, collate_fn=collate_fn) # Write your code here
dl_validation = DataLoader(SemevalDataset(split="validation"), shuffle=False, batch_size=validation_batch_size, collate_fn=collate_fn) # Write your code here
dl_test = DataLoader(SemevalDataset(split="test"), shuffle=False, batch_size=test_batch_size, collate_fn=collate_fn) # Write your code here

# TODO2: Construct your model
class MultiLabelModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Write your code here
        # Define what modules you will use in the model
        # Please use "google-bert/bert-base-uncased" model (https://huggingface.co/google-bert/bert-base-uncased)
        # Besides the base model, you may design additional architectures by incorporating linear layers, activation functions, or other neural components.
        # Remark: The use of any additional pretrained language models is not permitted.
        # Reference: https://io.traffine.com/en/articles/custom-bert-model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Fully-connected layer for classification task
        # Reference 1: https://www.kaggle.com/code/moeinshariatnia/simple-distilbert-fine-tuning-0-84-lb
        # Reference 2: https://pytorch-lightning.readthedocs.io/en/1.5.10/notebooks/lightning_examples/mnist-hello-world.html
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        )
        # Fully-connected layer for regression task
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.bert.config.hidden_size, regression)
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        # Write your code here
        # Forward pass
        # Reference: https://ithelp.ithome.com.tw/m/articles/10392188
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        pooled_output = outputs.pooler_output
        logits_fc1 = self.fc1(pooled_output)
        logits_fc2 = self.fc2(pooled_output)

        return logits_fc1, logits_fc2


# TODO3: Define your optimizer and loss function

model = MultiLabelModel().to(device)
# TODO3-1: Define your Optimizer
# Reference: https://ithelp.ithome.com.tw/m/articles/10392188
optimizer = AdamW(model.parameters(), lr=lr) # Write your code here

# TODO3-2: Define your loss functions (you should have two)
# Write your code here
# Reference: https://wellsr.com/python/solving-classification-and-regression-problems-with-pytorch/
loss_classif = torch.nn.CrossEntropyLoss()
loss_regress = torch.nn.MSELoss()

# scoring functions
psr = load("pearsonr")
acc = load("accuracy")

best_score = 0.0
for ep in range(epochs):
    pbar = tqdm(dl_train)
    pbar.set_description(f"Training epoch [{ep+1}/{epochs}]")
    model.train()
    # TODO4: Write the training loop
    # Write your code here
    # train your model
    # Reference: https://pyimagesearch.com/2022/08/17/multi-task-learning-and-hydranets-with-pytorch/
    for data_batch, label_relatedness_score, label_entailment_judgement in pbar : # Gemini help me to introduce this tuple unpacking idea, but i wrote the code by myself
        data_batch = data_batch.to(device=device)
        label_relatedness_score = label_relatedness_score.to(device=device)
        label_entailment_judgement = label_entailment_judgement.to(device=device)

        # clear gradient
        optimizer.zero_grad()
        # forward pass
        classif_output, regress_output = model(data_batch.input_ids, data_batch.attention_mask, data_batch.token_type_ids)
        # compute loss
        classif_loss = loss_classif(classif_output, label_entailment_judgement)

        # Reference: https://github.com/FernandoLpz/Text-Classification-LSTMs-PyTorch/issues/9
        regress_loss = loss_regress(regress_output, label_relatedness_score.unsqueeze(1))
        loss = classif_loss + regress_loss

        # back-propagation
        loss.backward()
        # model optimization
        optimizer.step()

    pbar = tqdm(dl_validation)
    pbar.set_description(f"Validation epoch [{ep+1}/{epochs}]")
    model.eval()
    # TODO5: Write the evaluation loop
    # Write your code here
    # Evaluate your model
    with torch.no_grad():
        # Reference: https://www.kaggle.com/code/martasprg/lecture-4-pytorch-training-loop
        val_preds_classif = []
        val_preds_regress = []
        val_labels_classif = []
        val_labels_regress = []

        for data_batch, label_relatedness_score, label_entailment_judgement in pbar:
            data_batch = data_batch.to(device=device)
            label_relatedness_score = label_relatedness_score.to(device=device)
            label_entailment_judgement = label_entailment_judgement.to(device=device)

            classif_output, regress_output = model(data_batch.input_ids, data_batch.attention_mask, data_batch.token_type_ids)
            
            val_preds_classif.append(classif_output)
            val_preds_regress.append(regress_output)

            val_labels_classif.append(label_entailment_judgement)
            val_labels_regress.append(label_relatedness_score)

    # Reference: https://discuss.pytorch.org/t/efficient-method-to-gather-all-predictions/8008/5
    y_true_classif = torch.cat(val_labels_classif)
    y_true_regress = torch.cat(val_labels_regress)

    output_classif = torch.cat(val_preds_classif)
    output_regress = torch.cat(val_preds_regress)

    # Reference 1: https://hackmd.io/@1x-Chen/rkfkwRLln
    # Reference 2: https://hackmd.io/@NCCU111356040/rkZiVbYcs
    y_preds_classif = torch.argmax(output_classif, dim=1)
    y_preds_regress = output_regress.squeeze()

    # Output all the evaluation scores (PearsonCorr, Accuracy)
    # Reference: https://huggingface.co/spaces/evaluate-metric/pearsonr
    pearson_corr = psr.compute(references=y_true_regress, predictions=y_preds_regress)['pearsonr'] # Write your code here
    print(f'Pearson: {pearson_corr}')
    # Reference: https://huggingface.co/spaces/evaluate-metric/accuracy
    accuracy = acc.compute(references=y_true_classif, predictions=y_preds_classif)['accuracy'] # Write your code here
    print(f'Accuracy: {accuracy}')
    # print(f"F1 Score: {f1.compute()}")

    if pearson_corr + accuracy > best_score:
        best_score = pearson_corr + accuracy
        torch.save(model.state_dict(), f'./saved_models/best_model.ckpt')

# Load the model
model = MultiLabelModel().to(device)
model.load_state_dict(torch.load(f"./saved_models/best_model.ckpt", weights_only=True))

# Test Loop
pbar = tqdm(dl_test, desc="Test")
model.eval()

# TODO6: Write the test loop
# Write your code here
# We have loaded the best model with the highest evaluation score for you
# Please implement the test loop to evaluate the model on the test dataset
# We will have 10% of the total score for the test accuracy and pearson correlation
with torch.no_grad():
    # Reference: https://www.kaggle.com/code/martasprg/lecture-4-pytorch-training-loop
    test_preds_classif = []
    test_preds_regress = []
    test_labels_classif = []
    test_labels_regress = []

    for data_batch, label_relatedness_score, label_entailment_judgement in pbar:
        data_batch = data_batch.to(device=device)
        label_relatedness_score = label_relatedness_score.to(device=device)
        label_entailment_judgement = label_entailment_judgement.to(device=device)

        classif_output, regress_output = model(data_batch.input_ids, data_batch.attention_mask, data_batch.token_type_ids)
        
        test_preds_classif.append(classif_output)
        test_preds_regress.append(regress_output)

        test_labels_classif.append(label_entailment_judgement)
        test_labels_regress.append(label_relatedness_score)

# Reference: https://discuss.pytorch.org/t/efficient-method-to-gather-all-predictions/8008/5
y_true_classif = torch.cat(test_labels_classif)
y_true_regress = torch.cat(test_labels_regress)

output_classif = torch.cat(test_preds_classif)
output_regress = torch.cat(test_preds_regress)

# Reference 1: https://hackmd.io/@1x-Chen/rkfkwRLln
# Reference 2: https://hackmd.io/@NCCU111356040/rkZiVbYcs
y_preds_classif = torch.argmax(output_classif, dim=1)
y_preds_regress = output_regress.squeeze()

# Output all the evaluation scores (PearsonCorr, Accuracy)
# Reference: https://huggingface.co/spaces/evaluate-metric/pearsonr
pearson_corr = psr.compute(references=y_true_regress, predictions=y_preds_regress)['pearsonr'] # Write your code here
print(f'Pearson: {pearson_corr}')
# Reference: https://huggingface.co/spaces/evaluate-metric/accuracy
accuracy = acc.compute(references=y_true_classif, predictions=y_preds_classif)['accuracy'] # Write your code here
print(f'Accuracy: {accuracy}')