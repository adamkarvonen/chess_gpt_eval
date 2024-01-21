## NanoGPT Specific Information

To evaluate a model, run this command:

`pip install torch numpy transformers datasets tiktoken wandb tqdm`

Download a model from: https://huggingface.co/adamkarvonen/chess_llms

And place it in `/nanogpt/out/`. `stockfish_16layers_ckpt_no_optimizer.pt` is the strongest model.
Then follow the remaining setup directions in the main README.

At the top of `main.py`, uncomment the nanogpt import. It is commented so we don't force a torch import by default. At the bottom of main.py, set `NANOGPT = False` and uncomment a NanoGptPlayer.