"""
Standalone script demonstrating the logic behind
https://arxiv.org/abs/2409.00009 with DuckDuckGo search.
"""
import json
import concurrent.futures
import os

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

load_dotenv()

import re
from typing import Any, Optional, NamedTuple, Tuple

import openai
from duckduckgo_search import DDGS
import pandas as pd

MAX_NUM_TURNS: int = 5
MAIN_AGENT_MODEL_NAME: str = "gpt-4o-mini"
MAX_SEARCH_RESULTS: int = 10
split = "train"
assert split in ["train", "val", "test"]

from pydantic import BaseModel


def google_search(query, result_total=10):
    """
    Conducts a Google search using the provided query and retrieves a specified number of results.

    :param query: The search query string.
    :param result_total: Total number of results desired.
    """

    def build_payload(query, start=1, num=10, **params):
        """
        Builds the payload for the Google Search API request.

        :param query: Search term.
        :param start: The index of the first result to return.
        :param num: Number of search results per request.
        :param params: Additional parameters for the API request.
        :return: Dictionary containing the API request parameters.
        """
        api_key = os.getenv("GOOGLE_CUSTOM_API_KEY")
        search_engine_ID = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
        payload = {
            'key': api_key,
            'q': query,
            'cx': search_engine_ID,
            'start': start,
            'num': num,
        }
        payload.update(params)
        return payload

    def make_request(payload):
        """
        Sends a GET request to the Google Search API and handles potential errors.

        :param payload: Dictionary containing the API request parameters.
        :return: JSON response from the API.
        """
        response = requests.get('https://www.googleapis.com/customsearch/v1', params=payload)
        if response.status_code != 200:
            raise Exception('Request failed with status code {}'.format(response.status_code))
        return response.json()

    items = []
    pages = (result_total + 9) // 10  # Ensuring we account for all pages including the last one which might be partial
    for i in range(pages):
        start_index = i * 10 + 1
        num_results = 10 if i < pages - 1 else result_total - (i * 10)
        payload = build_payload(query, start=start_index, num=num_results)
        response = make_request(payload)
        items.extend(response.get('items', []))

    return items


def ddg_search(query: str) -> str:
    """
    Invoke web search.

    Params:
        query: str

    Returns:
        Summarized response to search query.
    """
    res = ""
    results = DDGS().text(query, max_results=MAX_SEARCH_RESULTS * 2)

    results = [r for r in results if "politifact.com" not in r.get("href")][:MAX_SEARCH_RESULTS]
    for doc in results:
        res += f"Title: {doc['title']} Content: {doc['body'][:1600]}\n"

    # print("SEARCH is CALLED")
    completion = openai.chat.completions.create(
        messages=[{
            "role": "user",
            "content": f"You are a large language AI assistant built by Lepton AI. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable. \
            Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say ""information is missing on ..."" followed by the related topic, if the given context do not provide sufficient information. "
                       f"\nQuery: {query}\nSearch results: {res}",
        }],
        model=MAIN_AGENT_MODEL_NAME
    )

    response = completion.choices[0].message

    return response


class _KeywordExtractionOutput(NamedTuple):
    """Represent the part up to the matched string, and the match itself."""

    content_up_to_match: str
    matched_content: str


def _extract_search_query_or_none(
        assistant_response: str,
) -> Optional[_KeywordExtractionOutput]:
    """
    Try to extract "SEARCH: query\\n" request from the main agent response.

    Discards anything after the "query" part.

    Returns:
        _KeywordExtractionOutput if matched.
        None otherwise.
    """

    match = re.match(r"(.*SEARCH:\s+)(.+)([\n]+.+)", assistant_response)
    if match is None:
        return None

    return _KeywordExtractionOutput(
        content_up_to_match=match.group(1) + match.group(2),
        matched_content=match.group(2),
    )


def parse_factuality_output(output: str) -> dict:
    """
    Parses the formatted output enclosed within ```python tags into a dictionary.

    Parameters:
        output (str): The formatted string containing a Python dictionary.

    Returns:
        dict: A Python dictionary parsed from the string.
    """
    try:
        # Extract the content within ```python and ```
        start = output.find("```python")
        end = output.rfind("```")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("The provided output does not contain valid ```python ... ``` tags.")

        # Get the content inside the tags and strip extra whitespace
        content = output[start + len("```python"):end].strip()

        # Use `eval` to parse the Python dictionary string
        parsed_data = eval(content)

        # Check if it's a valid dictionary
        if not isinstance(parsed_data, dict):
            raise ValueError("Parsed content is not a dictionary.")

        return parsed_data

    except Exception as e:
        print(f"Error parsing the output: {e}")
        return {}


def parse_to_dict(enriched_questions_str: str) -> dict:
    """
    Parses a JSON-like string into a Python dictionary.

    Parameters:
        enriched_questions_str (str): The JSON-like string to be parsed.

    Returns:
        dict: Parsed dictionary of enriched questions.
    """
    # Step 1: Remove any markdown formatting like ```json or ```
    cleaned_str = re.sub(r"```json|```|```python", "", enriched_questions_str).strip()

    # Step 2: Parse the cleaned JSON string into a dictionary
    try:
        enriched_questions_dict = json.loads(cleaned_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}

    return enriched_questions_dict


def _extract_prediction_or_none(assistant_response: str) -> int | None:
    """
    Try to extract "Factuality: 1" (or 0, or 0.5) from main agent response.

    Response:
        Prediction value (as a string) if matched.
        None otherwise.
    """

    match = re.search(r"Factuality:\s(\d+\.?\d?)", assistant_response)
    if match is None:
        return None

    return int(float(match.group(1)))


def get_response(context, search_engine="ddg") -> tuple[Any, str]:
    response = openai.chat.completions.create(
        messages=context, model=MAIN_AGENT_MODEL_NAME
    )
    main_agent_message = response.choices[0].message.content
    assert main_agent_message is not None, (
        "Invalid Main Agent API response:",
        response,
    )

    # If search is requested in a message, truncate that message
    # up to the search request. (Discard anything after the query.)
    search_request_match = _extract_search_query_or_none(main_agent_message)
    if search_request_match is not None:
        if search_engine == 'ddg':
            search_response = ddg_search(search_request_match.matched_content)
        elif search_engine == 'google':
            search_response = google_search(search_request_match.matched_content)
        context += [
            {"role": "assistant", "content": search_request_match.content_up_to_match},
            {"role": "user", "content": f"Search result: {search_response}"},
        ]
        # continue
    else:
        context += [{"role": "assistant", "content": main_agent_message}]

    return context, main_agent_message


import re
import json
from typing import Optional


def parse_factuality_score(response: str) -> Optional[str]:
    """
    Parses a response string to extract the factuality score.
    Handles both clean JSON and code-block wrapped JSON.

    Parameters:
        response (str): The response string containing factuality information

    Returns:
        Optional[str]: The factuality score if found, None otherwise
    """
    # Step 1: Clean the string - remove code blocks and leading/trailing whitespace
    cleaned_str = re.sub(r'```python|```json|```', '', response).strip()

    try:
        # Step 2: Parse the JSON string
        # Fix common JSON formatting issues
        cleaned_str = re.sub(r',\s*}', '}', cleaned_str)  # Remove trailing commas
        cleaned_str = re.sub(r',\s*]', ']', cleaned_str)  # Remove trailing commas in arrays

        data_dict = json.loads(cleaned_str)

        # Step 3: Extract factuality score
        if "factuality" in data_dict:
            factuality_text = data_dict["factuality"]

            # Look for "Factuality: X" pattern
            match = re.search(r'Factuality:\s*(\d+(?:\.\d+)?)', factuality_text)
            if match:
                return match.group(1)

            # If no explicit score but contains a number, extract it
            number_match = re.search(r'\d+(?:\.\d+)?', factuality_text)
            if number_match:
                return number_match.group(0)

            # Return the full text if no number found
            return factuality_text

        else:
            print(f"No 'factuality' key found. Available keys: {list(data_dict.keys())}")
            return None

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        # Try to extract factuality score directly using regex if JSON parsing fails
        match = re.search(r'"factuality":\s*"[^"]*Factuality:\s*(\d+(?:\.\d+)?)"', cleaned_str)
        if match:
            return match.group(1)
        print(f"Could not parse response as JSON: {cleaned_str[:100]}...")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None


def process_factuality_evaluation(response: str) -> Optional[str]:
    """
    Process the factuality evaluation and handle the response.

    Parameters:
        response (str): The evaluation string to process

    Returns:
        Optional[str]: The factuality score if found, None otherwise
    """
    if not response or not response.strip():
        print("Empty or null response received")
        return None

    score = parse_factuality_score(response)
    if score is not None:
        return score

    print("Could not extract factuality score")
    return None


def perform_5w1h(statement, initial_enrich_query):
    enrichment_context: Any = [{"role": "user", "content": initial_enrich_query}]
    enrichment_context, enriched_questions = get_response(enrichment_context)
    # print(f"Enriched Questions: {enriched_questions}")

    # Step 2: Perform search for each question in enriched questions
    search_context = ""
    for category, questions in parse_to_dict(enriched_questions).items():
        for question in questions:
            # Summarize search results for each question
            search_summarization_query = f"""\
                You are given a user question, and please write clean, concise and accurate answer to the question.\
                Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. \
                Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

                If you cannot confidently provide a complete or accurate answer based on your knowledge, \
                 invoke the search engine tool to find additional information. To invoke the search engine, begin your query with the phrase \
                 "SEARCH: " followed by the query. Use the search engine as many times as needed to gather sufficient details \
                 and ensure your response is comprehensive.

                 If, after performing a search, the question still appears unanswerable, \
                 clearly state that the question cannot be answered 

                Statement: {statement + ", " + question}
            """
            search_summary_context: Any = [{"role": "user", "content": search_summarization_query}]
            search_summary_context, summarized_response = get_response(search_summary_context)
            # search_context[category].append(summarized_response)
            search_context += " " + summarized_response

            # If you cannot confidently provide a complete or accurate answer based on your knowledge, \
            #                 invoke the search engine tool to find additional information. To invoke the search engine, begin your query with the phrase \
            #                 "SEARCH: " followed by the query. Use the search engine as many times as needed to gather sufficient details \
            #                 and ensure your response is comprehensive.
            #
            #                 If, after performing a search, the question still appears unanswerable or misleading, \
            #                 clearly state that the question might be rooted in misinformation

            #  You are tasked with answering the given question as accurately as possible. \
            #                   Begin by assessing the question to determine if it is clear, factual, and complete. \
            #                   Note that the question might be misleading, incorrect, incomplete, or based on misinformation. \
            #                   If you identify such issues, state that the question is likely based on misinformation and provide an explanation.
            #
            #                     If you are confident in the question’s validity and can answer it using your current knowledge, \
            #                     proceed to provide a concise response.

            # print(f"Contextual Information: {search_context}")

        # Step 3: Analyze factuality based on gathered information
        factuality_query = f"""
                Based on the following gathered information, analyze the factuality of the original statement. \
                Summarize key supporting or contradicting details.
    
                Original Statement: {statement}
                Contextual Information: {search_context}
    
                Return the final output in dictionary:
                {{
                    "factuality": "True statement; Factuality: 1" or "False statement; Factuality: 0",
                    "summary": "A brief explanation",
                    "key_facts": ["Top supporting or contradicting facts"]
                }}
            """
        factuality_context = [{"role": "user", "content": factuality_query}]
        factuality_context, final_evaluation = get_response(factuality_context)

        factuality_score = process_factuality_evaluation(final_evaluation)
        # if factuality_score is not None:
        #     print(f"Factuality Score: {factuality_score}")
        return factuality_score


def process_row(row):
    statement = row['claim']

    initial_enrich_query = f"""\
           You have access to a language model. To enrich the context of the statement and reduce ambiguity, \
           transform the provided statement into six questions that can be helpful for validating the factuality of the statement, \
           each addressing one of the following aspects of the statement: \
           "when," "where," "what," "who," "why," and "how."

           Statement: {statement}

           After completing the transformations, output the final set of questions in strict dictionary format as shown:
           {{
               "when": [question related to time],
               "where": [question related to location],
               "what": [question related to the subject or event],
               "who": [question about people or entities involved]
               "why": [question regarding motives or reasons],
               "how": [question explaining methods or processes]
           }}
           """

    # If uncertain about any aspect, the model may propose multiple possibilities for that aspect.

    # "when": [list of questions related to time],
    #             "where": [list of questions related to location],
    #             "what": [list of questions related to the subject or event],
    #             "who": [list of questions about people or entities involved],
    #             "why": [list of questions regarding motives or reasons],
    #             "how": [list of questions explaining methods or processes]

    initial_search_query = f"""\
           You have access to a search engine tool. To invoke search, \
           begin your query with the phrase "SEARCH: ". You may invoke the search \
           tool as many times as needed. 

           Your task is to analyze the factuality of the given statement.

           Statement: {statement}

           After providing all your analysis steps, summarize your analysis \
           and state "True statement; Factuality: 1" if you think the statement \
           is factual, or "False statement; Factuality: 0" otherwise. \
           Please state the top important facts used for your decision without repeating the question following "Rationale: ".
           """

    return perform_5w1h(statement, initial_enrich_query), row['label']


def main(df):
    predictions = []
    labels = []
    # Using ThreadPoolExecutor for concurrent processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Creating future tasks for each row
        futures = [executor.submit(process_row, row) for index, row in df.iterrows()]
        # Collecting results as they are completed
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            predictions.append(future.result()[0])
            labels.append(future.result()[1])
    return predictions, labels


def filter_predictions_labels(predictions, labels):
    mismatch_indices = [i for i, (label, prediction) in enumerate(zip(labels, predictions)) if label != prediction]

    # Save the mismatch indices to a file
    with open("enrich_w_search_mismatch_indices.txt", "w") as file:
        for index in mismatch_indices:
            file.write(f"{index}\n")

    filtered_predictions_labels = [
        (int(pred), label) for pred, label in zip(predictions, labels)
        if pred in ['0', '1']
    ]

    try:
        filtered_predictions, filtered_labels = zip(*filtered_predictions_labels)
    except ValueError:
        filtered_predictions, filtered_labels = [], []

    return list(filtered_predictions), list(filtered_labels)


if __name__ == "__main__":
    dataset = "feverous"
    df = pd.read_parquet(f"hf://datasets/ComplexDataLab/Misinfo_Datasets/{dataset}/{dataset}_{split}.parquet")[
        ["claim", "label"]]

    label_counts = df['label'].value_counts()
    print("\nLabel counts:\n", label_counts)

    if dataset == "liar_new":
        df['label'] = df['label'].map({
            'false': 0,
            'pants-fire': 0,
            'barely-true': 0,
            'half-true': 1,
            'mostly-true': 1,
            'true': 1
        })
    elif dataset == "fever" or dataset == "feverous":
        df = df[df['label'].isin(['SUPPORTS', 'REFUTES', 'supports', 'refutes'])]

        df['label'] = df['label'].map({
            'SUPPORTS': 1,
            'supports': 1,
            'REFUTES': 0,
            'refutes': 0
        })

        supports = df[df['label'] == 1].sample(n=750, random_state=42)
        refutes = df[df['label'] == 0].sample(n=750, random_state=42)

        # Combine them into a balanced dataset
        balanced_df = pd.concat([supports, refutes])

        # Shuffle the coçmbined dataset
        df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    label_counts = df['label'].value_counts()
    print("\nLabel counts:\n", label_counts)

    predictions, labels = main(df)
    predictions, labels = filter_predictions_labels(predictions, labels)
    print(len(predictions), len(labels))

    # Calculating metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    # Printing the results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
