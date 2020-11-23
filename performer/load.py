import torch
from performer_pytorch import PerformerLM

# distil
gpt2_config = {
    'num_tokens': 50257,
    'max_seq_len': 1024,
    'dim': 768,
    'depth': 6,
    'heads': 12
}
model = PerformerLM(
    **gpt2_config,
    causal=True,
    reversible=True
)

model.load_state_dict(torch.load('./distilgpt2.pt'))

x = torch.randint(0, gpt2_config['num_tokens'],
                  (1, gpt2_config['max_seq_len']))
mask = torch.ones_like(x).bool()

output = model(x, mask=mask)  # (1, max_seq_len, vocab_size)
print('pass', output.shape)
