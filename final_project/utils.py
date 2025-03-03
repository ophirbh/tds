import openai
import json

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
openai.api_key =

def call_openai_llm(system_msg: str, user_msg: str) -> str:
    openai.api_key =
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Or use "gpt-3.5-turbo" for a more cost-effective option
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=150,  # Adjust as needed for the length of the answer
        temperature=0.7  # Controls randomness; adjust to taste
    )
    return response.choices[0].message.content.strip()


def get_structured_llm_response(messages, function_schema, function_call: str, model: str ="gpt-4o") -> str:
    openai.api_key =
    # Call the ChatCompletion API with function calling enabled.
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=[function_schema],
        function_call={"name": function_call}
    )

    # Extract the function call arguments (structured output).
    message = response["choices"][0]["message"]

    if "function_call" in message:
        arguments = json.loads(message["function_call"]["arguments"])
        missing_features = arguments.get("missing_features", [])
    else:
        # Fallback: try to parse a normal JSON response.
        try:
            missing_features = json.loads(message["content"]).get("missing_features", [])
        except Exception as e:
            print("Error parsing response:", e)
            missing_features = []

    print("Missing Features:", missing_features)
    return missing_features