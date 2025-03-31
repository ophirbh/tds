import json
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
OPEN_AI_KEY: str = os.environ.get('OPEN_AI_KEY')
client: OpenAI = OpenAI(api_key=OPEN_AI_KEY)

def find_missing_categorical_values(df: pd.DataFrame, regression_problem_statement: str,
                                                   categorical_values_upper_bound: int = 40,
                                                   additional_column_info: dict = None, model: str = "gpt-4o"):

    # Filter the dataframe to only include categorical columns.
    categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    categorical_features = [col for col in categorical_features if df[col].nunique() <= categorical_values_upper_bound]

    results = {}

    # Define the function schema (same for every feature)
    function_schema = {
        "name": "find_missing_values_of_categorical_features",
        "description": (
            "Return a list of important missing values of a categorical feature that are absent from the provided list of values. "
            "The categorical feature is described by its name, its list of values and a regression problem statement is provided. "
            "If no important missing values are found, return an empty list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "missing_values_of_categorical_features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of important missing values of a categorical feature relevant to the regression problem."
                }
            },
            "required": ["missing_values_of_categorical_features"]
        }
    }

    # Iterate feature by feature.
    for col in categorical_features:
        # Build the feature description with additional info if available.
        if additional_column_info is None or col not in additional_column_info:
            feature_description = f"{col}"
        else:
            feature_description = f"{col} ({additional_column_info[col]})"

        categorical_feature_values = df[col].unique()

        # Construct a user message for this feature.
        user_message = (
            f"I have this regression problem: '{regression_problem_statement}'. "
            f"I have a dataset with (among others) the categorical feature: {feature_description}. "
            f"This categorical feature has this list of unique values: {categorical_feature_values}. "
            "Please identify any important missing categorical features for this problem that are absent from the dataset. "
            "Please do not mention features that can be obtained by transforming the existing feature. "
            "If no important missing values are found, please return an empty list."
        )

        messages = [
            {"role": "system", "content": (
                "You are a data science expert who identifies important missing values of categorical features based on the provided regression problem."
            )},
            {"role": "user", "content": user_message}
        ]

        # Call the ChatCompletion API with function calling enabled.
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            functions=[function_schema],
            function_call={"name": "find_missing_values_of_categorical_features"}
        )

        message = response.choices[0].message

        # Try to parse the structured response.
        if message.function_call:
            try:
                arguments = json.loads(message.function_call.arguments)
                missing_values = arguments.get("missing_values_of_categorical_features", [])
            except Exception as e:
                print(f"Error parsing function response for feature {col}: {e}")
                missing_values = []
        else:
            # Fallback to parse a normal JSON response.
            try:
                missing_values = json.loads(message["content"]).get("missing_values_of_categorical_features", [])
            except Exception as e:
                print(f"Error parsing content response for feature {col}: {e}")
                missing_values = []

        results[col] = list(set(missing_values) - set(categorical_feature_values))

    return results
