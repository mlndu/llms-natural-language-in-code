import json
import random
from pathlib import Path
from typing import List, Dict
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput


DATASET_DIR = Path(__file__).resolve().parent / "data/code"


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


def split_data(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1):
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data


class CodeDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        label2id: Dict[str, int],
        max_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        code = item["code"]
        label = self.label2id[item["return_type"]]
        encodings = self.tokenizer(code, truncation=True, max_length=self.max_length, return_tensors="pt")
        encodings = {k: v.squeeze(0) for k, v in encodings.items()}
        encodings["labels"] = torch.tensor(label, dtype=torch.long)
        return encodings


class CodeBERTClassifier(nn.Module):
    def __init__(self, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        output_size = self.codebert.config.hidden_size
        hidden_size = output_size // 2
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(output_size, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None
    ):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0]
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.activation(x)
        logits = self.fc2(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def train_model(dataset_path: Path, output_dir: Path):
    print(f"\n{dataset_path.name}")

    data = load_jsonl(dataset_path)
    train_data, val_data, _ = split_data(data)

    label_set = sorted({item["return_type"] for item in data})
    label_id_map = {label: idx for idx, label in enumerate(label_set)}

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    train_ds = CodeDataset(train_data, tokenizer, label_id_map)
    val_ds = CodeDataset(val_data, tokenizer, label_id_map)

    model = CodeBERTClassifier(num_labels=len(label_set))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        num_train_epochs=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    trainer.save_model(str(output_dir))

    print(f"Model saved to {output_dir}\n")


def main():

    datasets = {
        "python_real": DATASET_DIR / Path("python_real.jsonl"),
        "python_combined": DATASET_DIR / "python_combined.jsonl",
    }

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    for name, path in datasets.items():
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        output_dir = models_dir / f"{name}_model"
        train_model(path, output_dir)


if __name__ == "__main__":
    main()
