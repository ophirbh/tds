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

    # Define the function schema (only used for models that support function calling).
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
            f"This categorical feature has these unique values: {list(categorical_feature_values)}. "
            "Please identify any important missing categorical values for this problem that are absent from the dataset. "
            "Do not include values that could be derived by transforming the existing ones."
        )

        messages = [
            {"role": "system", "content": (
                "You are a data science expert who identifies important missing values of categorical features based on the provided regression problem."
            )},
            {"role": "user", "content": user_message}
        ]

        # Call the ChatCompletion API differently based on model capability.
        if model == "gpt-4o":
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                functions=[function_schema],
                function_call={"name": "find_missing_values_of_categorical_features"}
            )
        else:
            # For models that do not support function calling.
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages
            )

        message = response["choices"][0]["message"]

        # Parse the structured response.
        missing_values = []
        if model == "gpt-4o" and "function_call" in message:
            try:
                arguments = json.loads(message["function_call"]["arguments"])
                missing_values = arguments.get("missing_values_of_categorical_features", [])
            except Exception as e:
                print(f"Error parsing function response for feature {col}: {e}")
                missing_values = []
        else:
            # Fallback for models without function calling: expect a plain JSON string in message["content"]
            try:
                parsed_content = json.loads(message["content"])
                missing_values = parsed_content.get("missing_values_of_categorical_features", [])
            except Exception as e:
                print(f"Error parsing content response for feature {col}: {e}")
                missing_values = []

        results[col] = missing_values
        print(f"Missing categorical features for '{col}':", missing_values)

    return results

if __name__ == '__main__':
    # Example usage (make sure the CSV path is correct)
    df = pd.read_csv("../data/processed_data/diamond_features.csv")
    res = find_missing_categorical_features_by_feature(
        df,
        "I wish to predict diamond prices according to the other features.",
        model="o1"  # Change this to "gpt-4o" if desired
    )
    print(res)
