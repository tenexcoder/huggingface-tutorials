import collections
import torch
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model_state = model.state_dict()

num_tokens = model.transformer.wte.num_embeddings
dim = model.transformer.wte.embedding_dim
max_seq_len = model.transformer.wpe.num_embeddings
depth = len(list(model.transformer.h.children()))

mingpt_state = collections.OrderedDict()

# emb
mingpt_state['tok_emb.weight'] = model_state['transformer.wte.weight']
mingpt_state['pos_emb.weight'] = model_state['transformer.wpe.weight']

# layers norm
mingpt_state['ln_f.weight'] = model_state['transformer.ln_f.weight']
mingpt_state['ln_f.bias'] = model_state['transformer.ln_f.bias']

attn_id = ['query', 'key', 'value']
for layer_idx in range(depth):
    from_layer = 'transformer.h.' + str(layer_idx)
    to_layer = 'blocks.' + str(layer_idx)

    # attn pre norm
    mingpt_state[to_layer + '.ln1.weight'] = \
        model_state[from_layer + '.ln_1.weight']
    mingpt_state[to_layer + '.ln1.bias'] = \
        model_state[from_layer + '.ln_1.bias']

    # load 'from' qkv weight and bias - qkv_w and qkv_b
    # then split into lists [q_w, k_w, v_w] and [q_b, k_b, v_b]
    qkv_w = model_state[from_layer +
                        '.attn.c_attn.weight'].split(dim, dim=1)
    qkv_b = model_state[from_layer +
                        '.attn.c_attn.bias'].split(dim, dim=0)

    # set 'to' qkv weight and bias - q_w, q_b, k_w, k_b. v_w, v_b
    for idx, (w, b) in enumerate(zip(qkv_w, qkv_b)):
        mingpt_state[to_layer + '.attn.' + attn_id[idx] + '.weight'] = w
        mingpt_state[to_layer + '.attn.' + attn_id[idx] + '.bias'] = b

    # attn projection
    mingpt_state[to_layer + '.attn.proj.weight'] = \
        model_state[from_layer + '.attn.c_proj.weight']
    mingpt_state[to_layer + '.attn.proj.bias'] = \
        model_state[from_layer + '.attn.c_proj.bias']

    # init mask
    mingpt_state[to_layer + '.attn.mask'] = \
        torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
            1, 1, max_seq_len, max_seq_len)

    # mlp pre norm
    mingpt_state[to_layer + '.ln2.weight'] = \
        model_state[from_layer + '.ln_2.weight']
    mingpt_state[to_layer + '.ln2.bias'] = \
        model_state[from_layer + '.ln_2.bias']

    # mlp - transpose 'from' to match 'to' shape
    # TODO: check why 'weight' shape isn't 1 to 1
    mingpt_state[to_layer + '.mlp.0.weight'] = \
        torch.einsum(
            'ij->ji',
            model_state[from_layer + '.mlp.c_fc.weight']
    )
    mingpt_state[to_layer + '.mlp.0.bias'] = \
        model_state[from_layer + '.mlp.c_fc.bias']
    mingpt_state[to_layer + '.mlp.2.weight'] = \
        torch.einsum(
            'ij->ji',
            model_state[from_layer + '.mlp.c_proj.weight']
    )
    mingpt_state[to_layer + '.mlp.2.bias'] = \
        model_state[from_layer + '.mlp.c_proj.bias']

# decoder head
mingpt_state['head.weight'] = \
    model_state['lm_head.weight']

torch.save(mingpt_state, './distilgpt2.pt')
print('done porting')
