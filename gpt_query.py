import openai
import tiktoken
import json
import os

# import replicate

# for hugging face inference endpoints for codellama
import requests

from typing import Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# system message is used in openai_request()
system_message = """Provide the next move in the chess game. Only provide the move, no move numbers."""

# dollars per 1k tokens, per openai.com/pricing
pricing_dict = {
    "gpt-4": 0.03,
    "gpt-4-0301": 0.03,
    "gpt-4-0613": 0.03,
    "gpt-3.5-turbo": 0.0015,
    "gpt-3.5-turbo-0301": 0.0015,
    "gpt-3.5-turbo-0613": 0.0015,
    "gpt-3.5-turbo-16k": 0.003,
    "babbage": 0.0005,
    "gpt-3.5-turbo-instruct": 0.0015,
}

MAX_TOKENS = 10

completion_models = [
    "gpt-3.5-turbo-instruct",
    "babbage",
    "davinci",
]


# tenacity is to handle anytime a request fails
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_gpt_response(
    prompt: str, model: str = "gpt-4", temperature: float = 0.0
) -> Optional[str]:
    try:
        messages = []
        # system message is used in openai_request()
        # system_message_dict = {
        #     "role": "system",
        #     "content": system_message,
        # }
        initial_message = {"role": "user", "content": prompt}
        messages.append(initial_message)

        record_messages(messages, model)

        # num_tokens = count_all_tokens(model, messages)
        # prompt_cost = get_prompt_cost(model, num_tokens)
        # print("prompt cost in $:", prompt_cost)

        if model in completion_models:
            response = get_completions_response(model, messages, temperature)
        elif model.startswith("gpt"):
            response = openai_chat_completion_request(model, messages, temperature)
        elif model.startswith("openrouter"):
            response = openrouter_request(model, messages, temperature)
        elif model.startswith("huggingface"):
            response = hugging_face_request(model, messages, temperature)
        elif model.startswith("replicate"):
            response = replicate_request(model, messages, temperature)
        else:
            raise Exception("Invalid model name")

        # response_cost = get_response_cost(model, count_tokens(model, response))
        # print("response cost in $:", response_cost)

        messages.append({"role": "assistant", "content": response})
        record_messages(messages, model)

        return response
    except Exception as e:
        print(f"Error while getting GPT response: {e}")
        return None


def openai_chat_completion_request(
    model: str, messages: list[dict], temperature: float
) -> str:
    system_message_dict = {
        "role": "system",
        "content": system_message,
    }
    messages.append(system_message_dict)
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    response = completion.choices[0].message.content
    return response


def openrouter_request(model: str, messages: list[dict], temperature: float) -> str:
    if temperature == 0:
        temperature = 0.001

    with open("gpt_inputs/openrouter_api_key.txt", "r") as f:
        openai.api_key = f.read().strip()

    openai.api_base = "https://openrouter.ai/api/v1"
    OPENROUTER_REFERRER = "https://github.com/adamkarvonen/nanoGPT"

    model = model.replace("openrouter/", "")

    completion = openai.ChatCompletion.create(
        model=model,
        headers={"HTTP-Referer": OPENROUTER_REFERRER},
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )
    response = completion.choices[0].message.content
    return response


def replicate_request(model: str, messages: list[dict], temperature: float) -> str:
    if temperature == 0:
        temperature = 0.001

    with open("gpt_inputs/replicate_api_key.txt", "r") as f:
        api_key = f.read().strip()
    os.environ["REPLICATE_API_TOKEN"] = api_key

    model = model.replace("replicate/", "")

    messages = translate_to_string_input(messages)

    output = replicate.run(
        model,
        input={
            "prompt": messages,
            "max_new_tokens": MAX_TOKENS,
            "temperature": temperature,
        },
    )

    # The meta/llama-2-7b model can stream output as it's running.
    response = ""
    # The predict method returns an iterator, and you can iterate over that output.
    for item in output:
        # https://replicate.com/meta/llama-2-7b/versions/527827021d8756c7ab79fde0abbfaac885c37a3ed5fe23c7465093f0878d55ef/api#output-schema
        response += item

    return response


def hugging_face_request(model: str, messages: list[dict], temperature: float) -> str:
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    messages = translate_to_string_input(messages)
    API_URL = "https://xxxxxxxx.us-east-1.aws.endpoints.huggingface.cloud"
    headers = {
        "Authorization": "Bearer xxxxx",
        "Content-Type": "application/json",
    }

    if temperature == 0:
        temperature = 0.001

    output = query(
        {
            "inputs": messages,
            "parameters": {"temperature": temperature, "max_new_tokens": MAX_TOKENS},
        }
    )

    return output[0]["generated_text"]


def translate_to_string_input(
    openai_messages: list[dict], roles_included: bool = False
):
    # Translate from OpenAI's dict to a single string input
    messages = []
    for message in openai_messages:
        if roles_included:
            messages.append(message["role"] + ": ")
        messages.append(message["content"])
    if roles_included:
        messages.append("assistant: ")
    return "\n".join(messages)


# for gpt-3 models and instruct models
def get_completions_response(
    model: str,
    messages: list[dict] | str,
    temperature: float,
    max_tokens: int = MAX_TOKENS,
) -> str:
    if not isinstance(messages, str):
        prompt = translate_to_string_input(messages, roles_included=False)
    else:
        prompt = messages

    completion = openai.Completion.create(
        model=model, temperature=temperature, prompt=prompt, max_tokens=max_tokens
    )

    response = completion.choices[0].text
    return response


def count_all_tokens(model: str, messages: list[dict[str, str]]) -> int:
    total_tokens = 0
    for message in messages:
        total_tokens += count_tokens(model, message["content"])
    return total_tokens


def count_tokens(model: str, prompt: str) -> int:
    if "gpt" not in model:
        model = "gpt-4"

    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(prompt))
    return num_tokens


def get_prompt_cost(model: str, num_tokens: int) -> float:
    # good enough for quick evals
    if model not in pricing_dict:
        return num_tokens * 0.001 * pricing_dict["gpt-4"]
    return num_tokens * 0.001 * pricing_dict[model]


def get_response_cost(model: str, num_tokens: int) -> float:
    # good enough for quick evals
    if model not in pricing_dict:
        return num_tokens * 0.001 * pricing_dict["gpt-4"]

    cost = num_tokens * 0.001 * pricing_dict[model]

    if model == "gpt-4":
        cost *= 2

    return cost


def record_messages(messages: list[dict], model: str):
    # create the conversation in a human-readable format
    conversation_text = ""
    for message in messages:
        conversation_text += message["content"]

    # write the conversation to the next available text file
    with open(f"gpt_outputs/transcript.txt", "w") as f:
        f.write(model + "\n\n")
        f.write(conversation_text)
