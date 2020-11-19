# Port Hugging Face weight to Performer

Note this is a work in progress, your mileage may vary. Initial tests yield promising results however not yet on par with vanilla transformers in terms of perplexity.

### Background

The goal of this repo is to show how we can port weights from the popular [Hugging Face transformers](https://github.com/huggingface/transformers) onto the SOTA [Performer](https://github.com/lucidrains/performer-pytorch). No better way to test the portability and efficiency of the Performer than to port the gpt2 weight to leverage its long context window of 1024 for finetuning.

As a smoke test we will be following their [fine-tuning a language model](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb#scrollTo=kswRMhPc3l-Q) on Wikitext 2 example. I have tried to replicate the training as close as possible alongside doing a 1:1 port of the vanilla model onto the Performer.

### Bounty

Open to pull requests to try to get the Performer model to be on par with the original Hugging Face exmaple which obtained ~3 perplexity by finetuning a pretrained distilgpt2 on Wikitext 2.
