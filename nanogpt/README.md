## NanoGPT Specific Information

To evaluate a model, run this command:

`pip install torch numpy transformers datasets tiktoken wandb tqdm`

Download a model from: https://huggingface.co/adamkarvonen/chess_llms

And place it in `/nanogpt/out/`. `stockfish_16layers_ckpt_no_optimizer.pt` is the strongest model.
Then follow the remaining setup directions in the main README.

At the top of `main.py`, uncomment the nanogpt import. It is commented so we don't force a torch import by default. At the bottom of main.py, set `NANOGPT = True` and uncomment a NanoGptPlayer.

To use the activations created in: https://github.com/adamkarvonen/chess_llm_interpretability

Create a `nanogpt/activations/` folder and copy your activations into there. Then, use this file to test your activations out: https://github.com/adamkarvonen/chess_gpt_eval/blob/grid_search/main_grid_search.py#L555-L620

Refer to the paper for the best coefficents and layer ranges to use. You will have to modify the string on line 558 to match your activations names.