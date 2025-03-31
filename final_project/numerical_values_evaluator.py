from openai import OpenAI
import json
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
OPEN_AI_KEY: str = os.environ.get('OPEN_AI_KEY')

def check_numerical_feature_validity(df: pd.DataFrame, regression_problem_statement: str,
                                               additional_column_info: dict = None, model: str = "gpt-4o"):
    client = OpenAI(api_key=OPEN_AI_KEY)

    # Filter the dataframe to only include numerical columns.
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()

    results = {}

    # Define the function schema for evaluating numerical features with reasoning.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "check_numerical_characteristics",
                "description": (
                    "Return a boolean value indicating if the numerical feature's current values make sense given the regression problem, "
                    "and provide a detailed explanation supporting your answer. For example, for a diamond price, you might expect the minimum "
                    "to be positive. The explanation should justify why the values make sense or not."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "values_make_sense": {
                            "type": "boolean",
                            "description": "A yes (true) or no (false) answer indicating if the numerical values make sense given the regression problem."
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "A detailed explanation supporting the boolean answer."
                        }
                    },
                    "required": ["values_make_sense", "reasoning"]
                }
            }
        }
    ]

    # Iterate over each numerical feature.
    for col in numerical_features:
        # Build the feature description with additional info if available.
        if additional_column_info is None or col not in additional_column_info:
            feature_description = f"{col}"
        else:
            feature_description = f"{col} ({additional_column_info[col]})"

        # Calculate descriptive statistics with additional measures.
        series = df[col]
        stats = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "variance": float(series.var()),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurt()),
            "25%": float(series.quantile(0.25)),
            "75%": float(series.quantile(0.75)),
            "IQR": float(series.quantile(0.75) - series.quantile(0.25))
        }

        # Construct the user message for this numerical feature.
        user_message = (
            f"I have this regression problem: '{regression_problem_statement}'. "
            f"I have a dataset with (among others) the numerical feature: {feature_description}. "
            f"This numerical feature has the following descriptive statistics: {json.dumps(stats)}. "
            "Please evaluate if the current values make sense given the context of the regression problem. "
            "Answer with true or false and include a detailed explanation for your answer."
        )

        messages = [
            {"role": "system", "content": (
                "You are a data science expert who evaluates whether a numerical feature's values make sense given a regression problem. "
                "Respond with a boolean answer (true or false) and include a detailed explanation supporting your answer."
            )},
            {"role": "user", "content": user_message}
        ]

        # Call the ChatCompletion API with function calling enabled using the new format
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "check_numerical_characteristics"}}
        )

        # Access the message differently in the new API
        message = response.choices[0].message

        # Parse the structured response.
        if message.tool_calls:
            try:
                tool_call = message.tool_calls[0]
                arguments = json.loads(tool_call.function.arguments)
                values_make_sense = arguments.get("values_make_sense", None)
                reasoning = arguments.get("reasoning", "")
            except Exception as e:
                print(f"Error parsing function response for feature {col}: {e}")
                values_make_sense = None
                reasoning = ""
        else:
            # Fallback to parsing a normal JSON response.
            try:
                parsed = json.loads(message.content)
                values_make_sense = parsed.get("values_make_sense", None)
                reasoning = parsed.get("reasoning", "")
            except Exception as e:
                print(f"Error parsing content response for feature {col}: {e}")
                values_make_sense = None
                reasoning = ""

        results[col] = {
            "values_make_sense": values_make_sense,
            "reasoning": reasoning
        }

    return results


def transform_skew(df, col):
    """
    Transforms the skewness of a column from right-skewed to left-skewed.

    This reflection is done using:
        new_value = (max_value + min_value) - original_value

    This transformation preserves a similar range without introducing
    negative values.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        col (str): The name of the column to transform.

    Returns:
        pd.DataFrame: The dataframe with an added column '<col>_transformed'
                      containing the transformed values.
    """
    # Calculate the min and max of the column.
    min_val = df[col].min()
    max_val = df[col].max()

    # Create a new column with the reflected values.
    df[col] = (max_val + min_val) - df[col]

    return df
