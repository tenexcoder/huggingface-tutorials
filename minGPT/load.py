import torch
from model import GPT

# distil
gpt2_config = {
    'vocab_size': 50257,
    'block_size': 1024,
    'n_embd': 768,
    'n_layer': 6,
    'n_head': 12
}
model = GPT({
    **gpt2_config,
    'embd_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'attn_pdrop': 0.1
})

model.load_state_dict(torch.load('./distilgpt2.pt'))

x = torch.randint(0, gpt2_config['vocab_size'],
                  (1, gpt2_config['block_size']))
mask = torch.ones_like(x).bool()

logits, loss = model(x)  # (1, block_size, vocab_size)
print('pass', logits.shape)
