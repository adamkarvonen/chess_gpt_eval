import openai
import gpt_query

openai.api_key_path = "gpt_inputs/api_key.txt"

with open("gpt_inputs/testing_input.txt", "r") as f:
    question = f.read()

model = "gpt-3.5-turbo"
model = "gpt-4"

num_tokens = gpt_query.count_tokens(model, question)

print("num input tokens", num_tokens)
print("cost", gpt_query.get_prompt_cost(model, num_tokens))

# system_message = """Your responses should be concise and strictly adhere to any formatting requirements."""

completion = openai.ChatCompletion.create(
    model=model,
    temperature=0.0,
    messages=[
        # {
        #     "role": "system",
        #     "content": system_message,
        # },
        {"role": "user", "content": question},
    ],
)

# print(completion.choices[0].message.content)

with open("gpt_test_output.txt", "w") as f:
    f.write(completion.choices[0].message.content)

response = gpt_query.get_completions_response(
    "gpt-3.5-turbo-instruct", question, 0.0, 200
)

print(response)
