"""
Sample from a trained model
"""
import os
import re
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from typing import Optional
from nanogpt.model import GPTConfig, GPT

BASE_DIR = "nanogpt/"


def add_activation_bias_to_state_dict(
    state_dict,
    device,
    activation_names: list[str],
    config: GPTConfig,
    activation_coefficient: float,
):
    activation_dir = "probe_derived"
    config.bias = True
    print(config)

    state_dict["transformer.ln_f.bias"] = torch.zeros_like(
        state_dict["transformer.ln_f.weight"]
    )

    for i in range(config.n_layer):
        layer_key = f"transformer.h.{i}"

        state_dict[f"{layer_key}.ln_1.bias"] = torch.zeros_like(
            state_dict[f"{layer_key}.ln_1.weight"]
        )
        state_dict[f"{layer_key}.ln_2.bias"] = torch.zeros_like(
            state_dict[f"{layer_key}.ln_2.weight"]
        )

        mlp_bias_shape = state_dict[f"{layer_key}.mlp.c_fc.weight"].shape[0]

        assert mlp_bias_shape == config.n_embd * 4

        state_dict[f"{layer_key}.mlp.c_fc.bias"] = torch.zeros(
            mlp_bias_shape, device=device
        )
        state_dict[f"{layer_key}.mlp.c_proj.bias"] = torch.zeros(
            config.n_embd, device=device
        )

        state_dict[f"{layer_key}.attn.c_attn.bias"] = torch.zeros(
            config.n_embd * 3, device=device
        )
        state_dict[f"{layer_key}.attn.c_proj.bias"] = torch.zeros(
            config.n_embd, device=device
        )

    for activation_name in activation_names:
        activation_state_dict = torch.load(
            f"nanogpt/activations/{activation_dir}/{activation_name}",
            map_location=device,
        )
        difference_vector = activation_state_dict["difference_vector"]
        difference_vector = difference_vector.detach()
        difference_vector *= activation_coefficient
        layer = activation_state_dict["layer"]
        # print(activation_state_dict.keys())

        # Add the difference vector to the attention bias
        state_dict[f"transformer.h.{layer}.mlp.c_proj.bias"] = difference_vector

    return state_dict, config


class NanoGptPlayer:
    def __init__(
        self,
        model_name: str,
        activation_name: Optional[str] = None,
        activation_coefficient: Optional[float] = None,
    ):
        self.model_name = model_name
        # -----------------------------------------------------------------------------

        init_from = "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
        out_dir = "out"  # ignored if init_from is not 'resume'
        input_dir = "addition"
        test_name = "test.txt"
        start = "12+44="  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
        num_samples = 1  # number of samples to draw
        max_new_tokens = 6  # number of tokens generated in each sample
        temperature = 0.01  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
        seed = 1337
        device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        # device = "cpu"
        dtype = "float16"  # 'float32' or 'bfloat16' or 'float16'
        compile = False  # use PyTorch 2.0 to compile the model to be faster
        exec(
            open(f"{BASE_DIR}configurator.py").read()
        )  # overrides from command line or config file
        # -----------------------------------------------------------------------------

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        device_type = (
            "cuda" if "cuda" in device else "cpu"
        )  # for later use in torch.autocast
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )

        # model
        if init_from == "resume":
            # init from a model saved in a specific directory
            ckpt_path = os.path.join(BASE_DIR, out_dir, self.model_name)
            ckpt_path = f"nanogpt/out/{self.model_name}"
            checkpoint = torch.load(ckpt_path, map_location=device)
            gptconf = GPTConfig(**checkpoint["model_args"])

            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

            if activation_name is not None:
                state_dict, gptconf = add_activation_bias_to_state_dict(
                    state_dict, device, activation_name, gptconf, activation_coefficient
                )
            model = GPT(gptconf)
            model.load_state_dict(state_dict)
            # model = torch.compile(model)
        elif init_from.startswith("gpt2"):
            # init from a given GPT-2 model
            model = GPT.from_pretrained(init_from, dict(dropout=0.0))

        model.eval()
        model.to(device)
        if compile:
            model = torch.compile(model)  # requires PyTorch 2.0 (optional)

        # look for the meta pickle in case it is available in the dataset folder
        load_meta = False
        if (
            init_from == "resume"
            and "config" in checkpoint
            and "dataset" in checkpoint["config"]
        ):  # older checkpoints might not have these...
            meta_path = os.path.join(BASE_DIR, "out", "meta.pkl")
            load_meta = os.path.exists(meta_path)
        if load_meta:
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            # TODO want to make this more general to arbitrary encoder/decoder schemes
            stoi, itos = meta["stoi"], meta["itos"]
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: "".join([itos[i] for i in l])
        else:
            # ok let's assume gpt-2 encodings by default
            print("No meta.pkl found, assuming GPT-2 encodings...")
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode = lambda l: enc.decode(l)

        self.encode = encode
        self.decode = decode
        self.model = model
        self.ctx = ctx
        self.device = device

    def get_nanogpt_response(self, game_state: str, temperature: float) -> str:
        num_samples = 1  # number of samples to draw
        top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
        max_new_tokens = 10

        # Remove ["stockfish elo xxx"]\n["stockfish elo xxx"]\n\n from game_state
        # nanogpt was trained only on pgn transcripts
        game_state = game_state.split("\n\n")[1].strip()

        # Nanogpt was trained on pgn transcripts of this format: 1.e4 e5 2.Nf3 (not 1. e4 e5 2. Nf3)
        # I did this to save on tokens
        # We remove the space after the move number to match the training data
        game_state = re.sub(r"(\d+\.) ", r"\1", game_state)

        game_state = ";" + game_state

        # print("game_state", game_state)

        start_ids = self.encode(game_state)

        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
        with torch.no_grad():
            with self.ctx:
                for k in range(num_samples):
                    y = self.model.generate(
                        x, max_new_tokens, temperature=temperature, top_k=top_k
                    )

                    model_response = self.decode(y[0].tolist())

        # print("model_response", model_response)
        # model_response includes the input string
        model_response = model_response[len(game_state) :]
        if ";" in model_response:
            model_response = model_response.split(";")[0]
        return model_response

    def get_move_from_response(self, response: str) -> str:
        # Parse the response to get only the first move
        moves = response.split()
        if not moves:
            return ""
        first_move = moves[0]

        return first_move

    def get_move(self, board: str, game_state: str, temperature: float) -> str:
        completion = self.get_nanogpt_response(game_state, temperature)
        return self.get_move_from_response(completion)

    def get_config(self) -> dict:
        return {"model": self.model_name}
