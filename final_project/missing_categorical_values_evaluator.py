import openai, json
import pandas as pd

def find_missing_categorical_features_by_feature(df: pd.DataFrame, regression_problem_statement: str,
                                                 categorical_values_upper_bound: int = 40,
                                                 additional_column_info: dict = None, model: str = "gpt-4o"):

    openai.api_key =

    # Filter the dataframe to only include categorical columns.
    categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    categorical_features = [col for col in categorical_features if df[col].nunique() <= categorical_values_upper_bound]

    results = {}

    # Define the function schema (same for every feature)
    function_schema = {
        "name": "find_missing_values_of_categorical_features",
        "description": (
            "Return a list of important missing values of a categorical feature that are absent from the provided list of values. "
            "The categorical feature is described by its name, its list of values and a regression problem statement is provided."
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
            f"This categorical feature has this list of unique values: {categorical_feature_values}"
            "Please identify any important missing categorical features for this problem that are absent from the dataset. "
            "Please do not mention features that can be obtained by transforming the existing feature."
        )

        messages = [
            {"role": "system", "content": (
                "You are a data science expert who identifies important missing values of categorical features based on the provided regression problem."
            )},
            {"role": "user", "content": user_message}
        ]

        # Call the ChatCompletion API with function calling enabled.
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=[function_schema],
            function_call={"name": "find_missing_values_of_categorical_features"}
        )

        message = response["choices"][0]["message"]

        # Try to parse the structured response.
        if "function_call" in message:
            try:
                arguments = json.loads(message["function_call"]["arguments"])
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

        results[col] = missing_values
        print(f"Missing categorical features for '{col}':", missing_values)

    return results

if __name__ == '__main__':
    df = pd.read_csv("../data/processed_data/diamond_features.csv")
    res = find_missing_categorical_features_by_feature(df, "I wish to predict diamond prices according to the other features.")
    print(res)
# Example usage:
# df = pd.DataFrame({
#     'color': ['red', 'blue', 'green'],
#     'type': ['A', 'B', 'A'],
#     'value': [1, 2, 3]  # non-categorical, will be ignored
# })
# regression_problem = "Predict the sales price of a product."
# additional_info = {'color': 'The primary color of the product', 'type': 'Product category'}
# missing_results = find_missing_categorical_features_by_feature(df, regression_problem, additional_info)
# print(missing_results)
