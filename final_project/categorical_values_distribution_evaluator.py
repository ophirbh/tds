import openai, json
import pandas as pd

def assess_distribution_of_categorical_features_by_feature(df: pd.DataFrame, regression_problem_statement: str,
                                                             categorical_values_upper_bound: int = 40,
                                                             additional_column_info: dict = None, model: str = "gpt-4o"):
    openai.api_key =

    # Filter the dataframe to only include categorical columns.
    categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    categorical_features = [col for col in categorical_features if df[col].nunique() <= categorical_values_upper_bound]

    results = {}

    # Define the function schema for assessing frequency distributions.
    function_schema = {
        "name": "assess_distribution_of_categorical_feature",
        "description": (
            "Return an assessment of the frequency distribution of a categorical feature. "
            "The feature is described by its name, its list of unique values, and their observed frequencies, "
            "as well as a regression problem statement. The response should include: "
            "1) a detailed textual assessment of whether the distribution makes sense or if there are any anomalies, "
            "and 2) a short answer (true or false) indicating if the distribution is logical given the problem."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "distribution_assessment": {
                    "type": "string",
                    "description": "A textual assessment of whether the frequency distribution of the categorical feature makes sense, including potential anomalies."
                },
                "is_distribution_logical": {
                    "type": "boolean",
                    "description": "A short answer indicating if the distribution seems logical (true) or not (false) given the regression problem."
                }
            },
            "required": ["distribution_assessment", "is_distribution_logical"]
        }
    }

    # Iterate feature by feature.
    for col in categorical_features:
        # Build the feature description with additional info if available.
        if additional_column_info is None or col not in additional_column_info:
            feature_description = f"{col}"
        else:
            feature_description = f"{col} ({additional_column_info[col]})"

        # Get unique values and their frequencies.
        frequency_distribution = df[col].value_counts().to_dict()

        # Construct a user message for this feature.
        user_message = (
            f"I have this regression problem: '{regression_problem_statement}'. "
            f"I have a dataset with (among others) the categorical feature: {feature_description}. "
            f"This feature has the following observed frequency distribution: {frequency_distribution}. "
            "Please assess whether this distribution makes sense given the problem. "
            "Provide a detailed explanation of any potential anomalies (such as skewness, missing expected categories, "
            "or unusually low frequency values) and also give a short answer (true or false) indicating if the distribution is logical."
        )

        messages = [
            {"role": "system", "content": (
                "You are a data science expert who evaluates the frequency distributions of categorical features "
                "for regression problems and provides insights on their adequacy and potential issues."
            )},
            {"role": "user", "content": user_message}
        ]

        # Call the ChatCompletion API with function calling enabled.
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=[function_schema],
            function_call={"name": "assess_distribution_of_categorical_feature"}
        )

        message = response["choices"][0]["message"]

        # Try to parse the structured response.
        if "function_call" in message:
            try:
                arguments = json.loads(message["function_call"]["arguments"])
                assessment = arguments.get("distribution_assessment", "")
                logical_flag = arguments.get("is_distribution_logical", False)
            except Exception as e:
                print(f"Error parsing function response for feature {col}: {e}")
                assessment = ""
                logical_flag = False
        else:
            # Fallback to parse a normal JSON response.
            try:
                parsed_response = json.loads(message["content"])
                assessment = parsed_response.get("distribution_assessment", "")
                logical_flag = parsed_response.get("is_distribution_logical", False)
            except Exception as e:
                print(f"Error parsing content response for feature {col}: {e}")
                assessment = ""
                logical_flag = False

        results[col] = {
            "assessment": assessment,
            "is_distribution_logical": logical_flag
        }
        print(f"Distribution assessment for '{col}':", assessment)
        print(f"Is the distribution logical for '{col}'?:", logical_flag)

    return results

if __name__ == '__main__':
    df = pd.read_csv("../data/processed_data/diamond_features.csv")
    res = assess_distribution_of_categorical_features_by_feature(
        df,
        "I wish to predict diamond prices based on the other features."
    )
    print(res)