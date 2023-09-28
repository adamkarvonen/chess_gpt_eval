from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

from typing import Optional


# There are a couple non optimal parts of this code:
# 1. It doesn't inherit the Player class in main.py, which throws type checking errors
# 2. get_move_from_response() is duplicated from main.py
# However, I didn't want to add clutter and major dependencies like torch, peft, and transformers
# to those not using this class. So, this was my compromise.
class BaseLlamaPlayer:
    def __init__(
        self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, model_name: str
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name

    def get_llama_response(self, game_state: str, temperature: float) -> Optional[str]:
        prompt = game_state
        tokenized_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        result = self.model.generate(
            **tokenized_input, max_new_tokens=10, temperature=temperature
        ).to("cpu")
        input_ids_tensor = tokenized_input["input_ids"]
        # transformers generate() returns <s> + prompt + output. This grabs only the output
        res_sliced = result[:, input_ids_tensor.shape[1] :]
        return self.tokenizer.batch_decode(res_sliced)[0]

    def get_move_from_response(self, response: Optional[str]) -> Optional[str]:
        if response is None:
            return None

        # Parse the response to get only the first move
        moves = response.split()
        first_move = moves[0] if moves else None

        return first_move

    def get_move(
        self, board: str, game_state: str, temperature: float
    ) -> Optional[str]:
        completion = self.get_llama_response(game_state, temperature)
        return self.get_move_from_response(completion)

    def get_config(self) -> dict:
        return {"model": self.model_name}


class LocalLlamaPlayer(BaseLlamaPlayer):
    def __init__(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=0
        ).to("cuda")
        super().__init__(tokenizer, model, model_name)


class LocalLoraLlamaPlayer(BaseLlamaPlayer):
    def __init__(self, base_model_id: str, adapter_model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
        model = (
            PeftModel.from_pretrained(base_model, adapter_model_path)
            .merge_and_unload()
            .to("cuda")
        )

        super().__init__(tokenizer, model, adapter_model_path)
