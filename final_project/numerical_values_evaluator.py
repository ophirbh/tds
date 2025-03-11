import openai
import json
import pandas as pd


def check_numerical_characteristic_consistency(df: pd.DataFrame, regression_problem_statement: str,
                                                 additional_column_info: dict = None, model: str = "gpt-4o"):
    openai.api_key =

    # Filter the dataframe to only include numerical columns.
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()

    results = {}

    # Define the function schema for evaluating numerical features with reasoning.
    function_schema = {
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
            "min": series.min(),
            "max": series.max(),
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "variance": series.var(),
            "skewness": series.skew(),
            "kurtosis": series.kurt(),
            "25%": series.quantile(0.25),
            "75%": series.quantile(0.75),
            "IQR": series.quantile(0.75) - series.quantile(0.25)
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

        # Call the ChatCompletion API with function calling enabled.
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=[function_schema],
            function_call={"name": "check_numerical_characteristics"}
        )

        message = response["choices"][0]["message"]

        # Parse the structured response.
        if "function_call" in message:
            try:
                arguments = json.loads(message["function_call"]["arguments"])
                values_make_sense = arguments.get("values_make_sense", None)
                reasoning = arguments.get("reasoning", "")
            except Exception as e:
                print(f"Error parsing function response for feature {col}: {e}")
                values_make_sense = None
                reasoning = ""
        else:
            # Fallback to parsing a normal JSON response.
            try:
                parsed = json.loads(message["content"])
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
        print(f"Feature '{col}': Values make sense: {values_make_sense}\nReasoning: {reasoning}\n")

    return results


if __name__ == '__main__':
    df = pd.read_csv("../data/processed_data/diamond_features.csv")
    res = check_numerical_characteristic_consistency(
        df,
        "I wish to predict diamond prices according to the other features.",
        additional_column_info={"price": "The sale price of the diamond", "carat": "Weight of the diamond"}
    )
    print(res)