import openai
import json
import pandas as pd

def find_missing_features(df: pd.DataFrame, regression_problem_statement: str, additional_column_info: dict = None, model: str ="gpt-4o"):
    openai.api_key =
    # Get the features (possibly with additional information) and example values.
    if additional_column_info is None:
        dataset_features = {col: df[col].iloc[0] for col in df.columns}
    else:
        dataset_features = {
            f"{col} ({additional_column_info[col]})" if col in additional_column_info else col: df[col].iloc[0]
            for col in df.columns
        }

    # Construct the user message.
    user_message = (
        f"I have a dataset with the following features and example values: {dataset_features}. "
        f"My regression problem is: '{regression_problem_statement}'. "
        "Please identify any important features that are missing (if there are such) from this dataset. "
        "Please don't mention features that can be obtained by transforming existing features."
    )

    # Define the function schema for finding missing features.
    function_schema = {
        "name": "find_missing_features",
        "description": (
            "Return a list of important missing features that are absent from the provided dataset. "
            "The dataset is described by feature names with example values, and a regression problem statement is provided."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "missing_features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of important missing features relevant to the regression problem."
                }
            },
            "required": ["missing_features"]
        }
    }

    # Create messages with a system message for context and the user query.
    messages = [
        {"role": "system", "content": (
            "You are a data science expert who identifies important missing features in a dataset "
            "based on the provided regression problem."
        )},
        {"role": "user", "content": user_message}
    ]

    # Call the ChatCompletion API with function calling enabled.
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=[function_schema],
        function_call={"name": "find_missing_features"}
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

if __name__ == '__main__':
    df = pd.read_csv("../data/data_round.csv")
    additional_column_info = {'Weight' : 'In Carat'}
    additional_column_info[2] = "Weight in Carat"
    missing_features = find_missing_features(df, "I wish to predict diamond prices according to the other features.", additional_column_info)
    print(missing_features)