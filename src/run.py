import json
import concurrent.futures
import os
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import openai
from duckduckgo_search import DDGS
import pandas as pd
from prompts import PromptManager

# Constants
load_dotenv()
MAX_NUM_TURNS = 5
MAIN_AGENT_MODEL_NAME = "gpt-4o-mini"
MAX_SEARCH_RESULTS = 10
DEBUG = False
split = "test"
assert split in ["train", "val", "test"]


def google_search(query, result_total=10):
    """Execute Google Custom Search API query."""
    api_key = os.getenv("GOOGLE_CUSTOM_API_KEY")
    search_engine_ID = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
    
    items = []
    pages = (result_total + 9) // 10
    for i in range(pages):
        start_index = i * 10 + 1
        payload = {
            'key': api_key,
            'q': query,
            'cx': search_engine_ID,
            'start': start_index,
            'num': min(10, result_total - (i * 10))
        }
        response = requests.get('https://www.googleapis.com/customsearch/v1', params=payload)
        if response.status_code != 200:
            raise Exception(f'Request failed with status code {response.status_code}')
        items.extend(response.json().get('items', []))
    return items

def ddg_search(query: str) -> str:
    """Execute DuckDuckGo search and summarize results."""
    result = ""
    results = DDGS().text(query, max_results=MAX_SEARCH_RESULTS * 2)
    results = [r for r in results if "politifact.com" not in r.get("href")][:MAX_SEARCH_RESULTS]

    for doc in results:
        result += f"Title: {doc['title']} Content: {doc['body'][:2400]}\n"

    # print("SEARCH is CALLED")
    completion = openai.chat.completions.create(
        messages=[{
            "role": "user",
            "content": PromptManager.get_search_summarization_prompt(query, result)
        }],
        model=MAIN_AGENT_MODEL_NAME
    )
    return completion.choices[0].message.content

def get_response(context, search_engine="none"):
    """Get response from OpenAI with optional search capability."""
    
    # main agent
    response = openai.chat.completions.create(
        messages=context, 
        model=MAIN_AGENT_MODEL_NAME
    )
    message = response.choices[0].message.content

    search_match = re.match(r"(.*SEARCH:\s+)(.+)([\n]+.+)", message)
    if search_match:
        query = search_match.group(2)
        if search_engine == 'ddg':
            context.extend([
                {"role": "assistant", "content": search_match.group(1) + query},
                {"role": "user", "content": f"Search result: {ddg_search(query)}"},
            ])
        elif search_engine == 'google':
            context.extend([
                {"role": "assistant", "content": search_match.group(1) + query},
                {"role": "user", "content": f"Search result: {google_search(query)}"},
            ])
        elif search_engine == 'none':
            # search_result = "Search engine is disabled."
            pass
        else:
            raise ValueError("Invalid search engine")
    else:
        context.append({"role": "assistant", "content": message})

    if DEBUG:
        print(context[-1]["content"])

    return context, message

def evaluate_factuality(statement, search_engine):
    """Evaluate factuality of a statement using enriched context."""
    enrich_query = PromptManager.get_enrichment_prompt(statement)
    context = [{"role": "user", "content": enrich_query}]
    search_context = ""
    
    # Get enriched questions and gather evidence
    context, questions = get_response(context, search_engine)
    questions_dict = json.loads(re.sub(r"```\w*\n|```", "", questions).strip())
    
    for category, questions in questions_dict.items():
        for question in questions:
            context, response = get_response([{
                "role": "user", 
                "content": f"Answer this question: {statement}, {question}"
            }])
            search_context += " " + response

    factuality_query = PromptManager.get_factuality_analysis_prompt(statement, search_context)
    context, evaluation = get_response([{"role": "user", "content": factuality_query}])
    match = re.search(r'Factuality:\s*(\d+)', evaluation)
    return match.group(1) if match else None

def process_dataset(df, search_engine):
    """Process dataset with parallel execution."""
    
    if DEBUG:
        return [(evaluate_factuality(row['claim'], search_engine), row['label']) for _, row in df.iterrows()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(lambda row: (evaluate_factuality(row['claim'], search_engine), row['label']), row) for _, row in df.iterrows()]
        results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures))]

    predictions, labels = zip(*[(int(p), l) for p, l in results if p in ['0', '1']])
    return list(predictions), list(labels)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate factuality of statements.")
    parser.add_argument("dataset", type=str, default="feverous", help="Dataset to evaluate")
    parser.add_argument("search_engine", type=str, default="none", help="Search engine to use")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    
    # dataset = "feverous"
    dataset = args.dataset
    search_engine = args.search_engine
    DEBUG = args.debug

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

    # predictions, labels = main(df)
    if DEBUG:
        df = df.sample(n=10, random_state=42)

    predictions, labels = process_dataset(df, search_engine=search_engine)
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
