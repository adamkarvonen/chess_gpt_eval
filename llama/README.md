## Llama Specific Information

Note: Llama is usable, but it requires much more compute to finetune Llama7B than to train a 50M parameter nanogpt. I have not invested the compute to train a Llama7B to play chess well. If you were to do so, you would probably want a 50 / 50 mix of chess data and something like OpenWebText so Llama retains its natural language abilities.

Install all dependencies at the top of `llama_module`.py.

At the top of `main.py`, uncomment the llama import. At the bottom of `main.py`, create a LocalLlama player. You will have to create your own model and specify the path to it.