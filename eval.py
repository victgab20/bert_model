
import pandas as pd
import torch
import random
import numpy as np
import nltk
nltk.download('punkt')

import tqdm
from nltk import tokenize

from transformers import BertTokenizer
from transformers import BertModel

import torchtext
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score


path = r"dataset"

class TextDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']

        # Tratamento de NaN ou valores numéricos no texto
        if isinstance(text, float) or pd.isna(text):
            text = ""

        # Tokenização com BERT
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Pegando o rótulo da base de dados
        label = torch.tensor(self.data.iloc[idx]['helpfulness'], dtype=torch.float)
        label_polarity = torch.tensor(self.data.iloc[idx]['stars'], dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # Remove a dimensão extra
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label_utility": label,
            "label_polarity": label_polarity
        }

word_to_index = {"<pad>": 0}  # Inicializa o dicionário com o token de padding

def process_splits(path, BATCH_SIZE, device):
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

    # Criando os datasets
    #train_dataset = TextDataset(path + '/balanced_train_filmes.csv', tokenizer)
    #valid_dataset = TextDataset(path + '/balanced_dev_filmes.csv', tokenizer)
    test_dataset = TextDataset(path + '/balanced_test_filmes.csv', tokenizer)

    # Criando os DataLoaders com a função `collate_batch`
    #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader
    #return train_loader, valid_loader, test_loader

# Defina os parâmetros antes de chamar a função
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# glove_path = f"{path}/glove_s300.txt"  # Defina o caminho correto

#train_iterator, valid_iterator, 
test_iterator = process_splits(path, BATCH_SIZE, device)


class MultiTaskBertModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_size=768):
        super(MultiTaskBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_polarity = nn.Linear(hidden_size, 1)  # Para a tarefa de polaridade
        self.fc_usefulness = nn.Linear(hidden_size, 1)  # Para a tarefa de utilidade

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Pega a saída [CLS] token

        pooled_output = self.dropout(pooled_output)

        # Saída para a tarefa de polaridade
        polarity_output = self.fc_polarity(pooled_output)

        # Saída para a tarefa de utilidade
        usefulness_output = self.fc_usefulness(pooled_output)

        return polarity_output, usefulness_output

def binary_accuracy(preds, y):

    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    total_predictions = []
    total_labels_usefulness = []
    total_labels_polarity = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(iterator, desc='evaluating...'):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_usefulness = batch["label_utility"].to(device)
            labels_polarity = batch["label_polarity"].to(device)

            # Passa pelo modelo
            polarity_predictions, usefulness_predictions = model(input_ids=input_ids, attention_mask=attention_mask)

            # Remove a dimensão extra
            predictions_usefulness = usefulness_predictions.squeeze(1)
            predictions_polarity = polarity_predictions.squeeze(1)

            predictions_usefulness = torch.sigmoid(usefulness_predictions).squeeze(1)
            predictions_polarity = torch.sigmoid(polarity_predictions).squeeze(1)

            # Salva previsões para avaliação posterior
            total_predictions.append({
                "usefulness": predictions_usefulness.tolist(),
                "polarity": predictions_polarity.tolist()
            })

            # Acumula rótulos para calcular as métricas
            total_labels_usefulness.append(labels_usefulness.cpu().numpy())
            total_labels_polarity.append(labels_polarity.cpu().numpy())

            # Calcula a perda separadamente
            loss_usefulness = criterion(predictions_usefulness, labels_usefulness)
            loss_polarity = criterion(predictions_polarity, labels_polarity)
            loss = loss_usefulness + loss_polarity  # Soma das duas perdas

            # Calcula acurácia separadamente e faz a média
            acc_usefulness = binary_accuracy(predictions_usefulness, labels_usefulness)
            acc_polarity = binary_accuracy(predictions_polarity, labels_polarity)
            acc = (acc_usefulness + acc_polarity) / 2  # Média das duas acurácias

            # Acumula perdas e acurácias
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    # Concatenar todas as previsões e rótulos para calcular as métricas
    all_predictions_usefulness = torch.tensor([item["usefulness"] for item in total_predictions]).flatten()
    all_predictions_polarity = torch.tensor([item["polarity"] for item in total_predictions]).flatten()

    all_labels_usefulness = torch.tensor(total_labels_usefulness).flatten()
    all_labels_polarity = torch.tensor(total_labels_polarity).flatten()

    # Calcular as métricas de desempenho
    f1_usefulness = f1_score(all_labels_usefulness, all_predictions_usefulness.round())
    precision_usefulness = precision_score(all_labels_usefulness, all_predictions_usefulness.round())
    recall_usefulness = recall_score(all_labels_usefulness, all_predictions_usefulness.round())

    f1_polarity = f1_score(all_labels_polarity, all_predictions_polarity.round())
    precision_polarity = precision_score(all_labels_polarity, all_predictions_polarity.round())
    recall_polarity = recall_score(all_labels_polarity, all_predictions_polarity.round())

    # Criar um DataFrame com as métricas
    metrics = {
        "Test Loss": [epoch_loss / len(iterator)],
        "Test Accuracy": [epoch_acc / len(iterator)],
        "F1 Usefulness": [f1_usefulness],
        "Precision Usefulness": [precision_usefulness],
        "Recall Usefulness": [recall_usefulness],
        "F1 Polarity": [f1_polarity],
        "Precision Polarity": [precision_polarity],
        "Recall Polarity": [recall_polarity]
    }

    df_metrics = pd.DataFrame(metrics)

    # Salvar os resultados em um arquivo CSV
    df_metrics.to_csv("evaluation_metrics.csv", index=False)

    return (
        epoch_loss / len(iterator),
        epoch_acc / len(iterator),
        f1_usefulness, precision_usefulness, recall_usefulness,
        f1_polarity, precision_polarity, recall_polarity,
        total_predictions
    )

# Carregar o modelo
model = MultiTaskBertModel()
model = model.to(device)
model.load_state_dict(torch.load(r'C:\Users\victo\Downloads\modelos/bert_best_filmes.pt'))
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

# Avaliar no conjunto de teste
test_loss, test_acc, f1_usefulness, precision_usefulness, recall_usefulness, f1_polarity, precision_polarity, recall_polarity, predictions = evaluate(model, test_iterator, criterion)

# Exibir resultados
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
print(f'Usefulness - F1: {f1_usefulness:.3f} | Precision: {precision_usefulness:.3f} | Recall: {recall_usefulness:.3f}')
print(f'Polarity - F1: {f1_polarity:.3f} | Precision: {precision_polarity:.3f} | Recall: {recall_polarity:.3f}')