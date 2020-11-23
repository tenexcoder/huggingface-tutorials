# replicating Fine-tuning a language model - huggingface/transformers/examples/language_modeling.ipynb
import pytorch_lightning as pl
from model import GPT
from lm_wrapper import AutoregressiveWrapper
import torch
from torch.nn import functional as F
from transformers.data.data_collator import default_data_collator
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

tokenizer = AutoTokenizer.from_pretrained('distilgpt2', use_fast=True)


def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = datasets.map(
    tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

block_size = 128*1


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)


batch_size = 6
train_dataloader = DataLoader(
    lm_datasets['train'],
    batch_size=batch_size,
    collate_fn=default_data_collator
)
val_dataloader = DataLoader(
    lm_datasets['train'],
    batch_size=batch_size,
    collate_fn=default_data_collator
)


class LitGpt(pl.LightningModule):

    def __init__(self, weight_decay, lr, pretrained=True):
        super().__init__()
        self.weight_decay = weight_decay
        self.lr = lr
        self.model = GPT({
            'vocab_size': 50257,
            'block_size': 1024,
            'n_embd': 768,
            'n_layer': 6,
            'n_head': 12,
            'embd_pdrop': 0.1,
            'resid_pdrop': 0.1,
            'attn_pdrop': 0.1
        })
        if pretrained:
            print('pretrained')
            self.model.load_state_dict(torch.load('./distilgpt2.pt'))
        # self.model = AutoregressiveWrapper(self.model)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def training_step(self, batch, batch_idx):
        lm_logits, _ = self(batch['input_ids'])
        # training_step defined the train loop. It is independent of forward
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = batch['input_ids'][..., 1:].contiguous()
        # print(batch['input_ids'])
        # print(shift_labels)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        return [optimizer], []


weight_decay = 0.01
lr = 2e-5
autoencoder = LitGpt(weight_decay, lr)
trainer = pl.Trainer(max_epochs=3)
trainer.fit(autoencoder, train_dataloader, val_dataloader)
