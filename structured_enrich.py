import json
import concurrent.futures
import os
import time
from typing import Any, Optional, Dict, List, Tuple
from enum import Enum
import pandas as pd
import requests
from tqdm import tqdm
import openai
from duckduckgo_search import DDGS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv
from structured_format import FactCheck, AnswersList, QuestionSet, SearchResultSummary

# Load environment variables
load_dotenv()


class SearchProvider(Enum):
    GOOGLE = "google"
    DUCKDUCKGO = "duckduckgo"


class Search:
    def __init__(
            self,
            provider: SearchProvider = SearchProvider.DUCKDUCKGO,
            max_results: int = 5,
            calls_per_second: float = 0.2  # 1 call per 5 seconds
    ):
        self.provider = provider
        self.max_results = max_results

    def _ddg_search(self, query: str) -> List[Dict[str, str]]:
        try:
            results = DDGS().text(query, max_results=self.max_results)
            return [r for r in results if "politifact.com" not in r.get("href")][:self.max_results]
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []

    def _google_search(self, query: str) -> List[Dict[str, Any]]:

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
            try:
                response = requests.get('https://www.googleapis.com/customsearch/v1', params=payload)
                if response.status_code != 200:
                    raise Exception('Request failed with status code {}'.format(response.status_code))
                return response.json()
            except Exception as e:
                print(f"Google search error: {e}")
                return []

        items = []
        pages = (
                            self.max_results + 9) // 10  # Ensuring we account for all pages including the last one which might be partial
        for i in range(pages):
            start_index = i * 10 + 1
            num_results = 10 if i < pages - 1 else self.max_results - (i * 10)
            payload = build_payload(query, start=start_index, num=num_results)
            response = make_request(payload)
            items.extend(response.get('items', []))
        return items

    def search(self, query: str) -> str:
        if self.provider == SearchProvider.DUCKDUCKGO:
            results = self._ddg_search(query)
            return "\n".join(f"Title: {r['title']} Content: {r['body'][:1600]}" for r in results)
        else:  # Google
            results = self._google_search(query)
            return "\n".join(f"Title: {r.get('title', '')} Content: {r.get('snippet', '')[:1600]}" for r in results)


class EvaluationMode(Enum):
    DIRECT_INTERNAL = "direct_internal"  # No decomposition, internal knowledge only
    DIRECT_SEARCH = "direct_search"  # No decomposition, with search
    DECOMP_INTERNAL = "decomp_internal"  # 5W1H decomposition, internal knowledge only
    DECOMP_SEARCH = "decomp_search"  # 5W1H decomposition with search


class FactualityEvaluator:
    def __init__(
            self,
            mode: EvaluationMode = EvaluationMode.DIRECT_INTERNAL,
            model_name: str = "gpt-4o-mini",
            max_search_results: int = 5,
            search_provider: SearchProvider = SearchProvider.DUCKDUCKGO,
    ):
        self.mode = mode
        self.model_name = model_name
        self.max_search_results = max_search_results
        self.search_client = Search(
            provider=search_provider,
            max_results=max_search_results,
        )

    def get_response(self, context, format=None) -> tuple[Any, dict]:
        """Get response from the language model."""
        response = openai.beta.chat.completions.parse(
            messages=context,
            model=self.model_name,
            response_format=format
        )
        main_agent_message = response.choices[0].message.content
        return context, self.parse_to_dict(main_agent_message)

    def call_search_engine(self, statement: str, query=None) -> dict:
        """Perform search with rate limiting."""
        if query is None:
            res = self.search_client.search(statement)
            search_result_summary_query = f"""
                        Given a statement find search results that is helpful for validating the claim, your task is to:

                        1. First, discard results that are off-topic or only tangentially related
                        2. Second, determine if the statement can be supported/ refuted based on the provided search results
                        3. If the question CANNOT be supported or refuted or information is missing:
                           - Immediately state "INSUFFICIENT INFORMATION" 
                           - Stop processing and do not attempt to draw conclusions

                        For statement that CAN be answered, provide the following in your output:

                        search_findings: Summary of relevant information found in the search results that support/ refute the statement,
                        relationship_to_statement: NA,
                        confidence_level: High/Medium/Low based on the quality and relevance of available information",


                        For questions that CANNOT be answered, format your output as follows:

                        search_findings: INSUFFICIENT_INFORMATION,
                        relationship_to_statement: NA,
                        confidence_level: NA

                        Guidelines:
                        - For any statements, don't make assumptions or draw conclusions
                        - Focus only on information directly related to the statement
                        - Distinguish between definitive facts and interpretations
                        - Note when search results are insufficient to draw conclusions and do not speculate when information is missing
                        - Highlight any contradictions in the search results
                        - Consider the reliability and recency of sources
                        - Flag any potentially outdated information

                        Example input:
                        Statement: "Geedam had a population of 45,025 in 2011"
                        Search Results: [Search result content would appear here]

                        Statement: {statement},
                        Search Results: {res}


                        Provide your output in the given format.
                        """
        else:
            res = self.search_client.search(query)

            search_result_summary_query = f"""
            Given a statement, a specific question about that statement, and search results related to the question, your task is to:
    
            1. First, discard results that are off-topic or only tangentially related
            2. Second, determine if the question can be answered based on the provided search results
            3. If the question CANNOT be answered or information is missing:
               - Immediately state "INSUFFICIENT INFORMATION" 
               - Stop processing and do not attempt to draw conclusions
            4. If the question CAN be answered, analyze the search results and provide your findings
            
            For questions that CAN be answered, provide the following in your output:
    
            search_findings: Summary of relevant information found in the search results that answers the question,
            relationship_to_statement: Briefly explain how these findings support or contradict the original statement,
            confidence_level: High/Medium/Low based on the quality and relevance of available information",
            
            
            For questions that CANNOT be answered, format your output as follows:
            
            search_findings: INSUFFICIENT_INFORMATION,
            relationship_to_statement: NA,
            confidence_level: NA
    
            Guidelines:
            - Make the distinction between unanswerable and answerable questions immediately clear
            - For unanswerable questions, don't make assumptions or draw conclusions
            - Focus only on information directly related to the question
            - Distinguish between definitive facts and interpretations
            - Note when search results are insufficient to draw conclusions and do not speculate when information is missing
            - Highlight any contradictions in the search results
            - Consider the reliability and recency of sources
            - Flag any potentially outdated information
            
            Example input:
            Statement: "Geedam had a population of 45,025 in 2011"
            Question: "What year is the population figure of 45,025 for Geedam reported from?"
            Search Results: [Search result content would appear here]
            
            Statement: {statement},
            Question: {query},
            Search Results: {res}
            
            
            Provide your output in the given format.
            """

        _, response = self.get_response([{"role": "user", "content": search_result_summary_query}], SearchResultSummary)
        return response

    def parse_to_dict(self, questions_str: str) -> dict:
        """Parse the 5W1H questions string into a dictionary."""
        cleaned_str = questions_str.replace("```json", "").replace("```python", "").replace("```", "").strip()
        try:
            return json.loads(cleaned_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing questions: {e}")
            return {}

    def generate_5w1h_questions(self, statement: str) -> Dict[str, List[str]]:
        """Generate the 5W1H questions for a statement."""
        initial_enrich_query = f"""
        You are a fact-checking assistant specializing in generating natural, search-friendly questions. \
        Transform the provided statement into six questions that will help validate its factuality. 
        Write questions as if you were typing them into a search engine - use natural language that real people would use when searching.

        You have access to a language model. To enrich the context of the statement and reduce ambiguity, \
        transform the provided statement into six questions that can be helpful for validating the factuality of the statement, \
        each addressing one of the following aspects of the statement: \
        "when," "where," "what," "who," "why," and "how."

        Statement: {statement}

        After completing the transformations, output the final set of questions to the given structure.
        """

        _, response = self.get_response([{"role": "user", "content": initial_enrich_query}], QuestionSet)
        return response

    def generate_fact_score(self, statement: str, search_context, unanswerable_count) -> tuple[dict, str]:
        factuality_query = f"""
                        You are a fact-checking assistant. Evaluate statements using both the provided contextual information AND \
                        your general knowledge. When these sources conflict, prioritize recent contextual information over internal knowledge.

                        Original Statement: {statement}
                        Contextual Information: {search_context}
                        Unanswerable Percentage: {unanswerable_count / 6}

                        Evaluation Rules:
                        1. If unanswerable percentage is above 51%, factuality score is automatically 0
                        2. Words like "may," "might," "could," "likely" in supporting evidence should be treated as insufficient proof
                        3. For statements about studies/reports, verify both:
                           - If the cited source made that specific claim
                           - If the claim aligns with verified facts from your knowledge
                        4. Mark as True (1) only if supported by either:
                           - Explicit evidence in the context
                           - Well-established facts from your knowledge
                        5. Any significant uncertainty requires a False (0) score

                        Please provide:
                        factuality: True statement or False statement
                        factuality_score: 1 or 0
                        summary: Explanation incorporating both provided evidence and relevant general knowledge
                        key_facts: Supporting evidence from both context and known facts

                        Return the final output in the given structure.
                    """

        _, final_evaluation = self.get_response([{"role": "user", "content": factuality_query}], FactCheck)

        return final_evaluation, statement

    def get_question_answer_internal(self, statement: str, question: str) -> dict:
        """Get answer for a specific question using internal knowledge only."""
        evaluation_prompt = f"""
        You are statement and a question related to the statement, and please write clean, concise and accurate answer to the question.\
        Identify if the question is answerable or not.
        If yes, answer the question in unbiased and professional tone based on your best knowledge.
        Do not give any information that is not related to the question, and do not repeat. \
        
        Statement: {statement}
        Question: {question}
        
        Provide the following: 
        search_findings: Summary of relevant information found in the search results that answers the question,
        relationship_to_statement: Briefly explain how these findings support or contradict the original statement,
        confidence_level: High/Medium/Low based on the quality and relevance of available information",
        
        
        For questions that CANNOT be answered, format your output as follows:
        
        search_findings: INSUFFICIENT_INFORMATION, if you do not have sufficient information.
        relationship_to_statement: NA,
        confidence_level: NA
        
        Return the list of answers in the given format.
        """

        _, response = self.get_response([{"role": "user", "content": evaluation_prompt}], SearchResultSummary)
        return response

    def evaluate_decomp_search(self, statement: str) -> tuple[dict, str] | None:
        """Evaluate using 5W1H decomposition with search capability."""
        try:
            # Generate 5W1H questions
            questions = self.generate_5w1h_questions(statement)
            search_context = ""
            unanswerable_count = 0
            for category, question in questions.items():
                answer = self.call_search_engine(statement, question)
                search_summary = answer["search_findings"]
                if search_summary != "INSUFFICIENT_INFORMATION":
                    search_context += " " + search_summary
                else:
                    unanswerable_count += 1

            return self.generate_fact_score(statement, search_context, unanswerable_count)

        except Exception as e:
            print(f"Error during 5W1H evaluation with search: {e}")
            return None

    def evaluate_decomp_internal(self, statement: str) -> tuple[dict, str] | None:
        """Evaluate using 5W1H decomposition with internal knowledge only."""
        try:
            # Generate 5W1H questions
            questions = self.generate_5w1h_questions(statement)
            # Get answers to each question using internal knowledge
            context = ""
            unanswerable_count = 0
            for category, question in questions.items():
                answer = self.get_question_answer_internal(statement, question)
                search_summary = answer["search_findings"]
                if search_summary != "INSUFFICIENT_INFORMATION":
                    context += " " + search_summary
                else:
                    unanswerable_count += 1

            return self.generate_fact_score(statement, context, unanswerable_count)

        except Exception as e:
            print(f"Error during 5W1H evaluation with internal knowledge: {e}")
            return None

    def evaluate_direct_search(self, statement: str) -> tuple[dict, str] | None:
        """Evaluate directly with search capability."""
        try:
            search_context = ""
            unanswerable_count = 0
            answer = self.call_search_engine(statement)
            search_summary = answer["search_findings"]
            if search_summary != "INSUFFICIENT_INFORMATION":
                search_context += " " + search_summary
            else:
                unanswerable_count += 1

            return self.generate_fact_score(statement, search_context, unanswerable_count)

        except Exception as e:
            print(f"Error during direct evaluation with search: {e}")
            return None

    def evaluate_direct_internal(self, statement: str) -> tuple[dict, str] | None:
        """Evaluate directly with internal knowledge only."""
        try:
            factuality_query = f"""
                            You are a misinformation detection assistant. Your task is to determine if a statement contains false or misleading information, regardless of intent. Consider all forms of misinformation including false claims, misrepresentation of real events/studies, out-of-context information, and misleading presentations of facts.
                            
                            Original Statement: {statement}
                            
                            Evaluation Rules:
                            1. Score True (1) for statements that:
                               - Present accurate information in its proper context
                               - Make claims that match their cited sources
                               - Draw reasonable conclusions from real events/data
                               - Express opinions/speculation clearly marked as such
                            
                            2. Score False (0) for statements that:
                               - Contain false claims or fabricated information
                               - Misrepresent real studies, events, or data
                               - Take real information out of context in a misleading way
                               - Make false attributions to sources
                               - Create misleading connections between real events
                               - Present speculation as established fact
                            
                            3. For source-attributed claims:
                               - First verify if the source exists and made a similar claim
                               - Then check if the statement accurately represents the source's claim

                            4. Any significant uncertainty requires a False (0) score

                            Please provide:
                            factuality: True statement or False statement
                            factuality_score: 1 or 0
                            summary: Explanation incorporating both provided evidence and relevant general knowledge
                            key_facts: Supporting evidence from both context and known facts

                            Return the final output in the given structure.
                        """

            _, final_evaluation = self.get_response([{"role": "user", "content": factuality_query}], FactCheck)
            # print(final_evaluation)
            return final_evaluation, statement

        except Exception as e:
            print(f"Error during direct internal evaluation: {e}")
            return None

    def evaluate(self, statement: str) -> Optional[str]:
        """Main evaluation method that uses the specified mode."""
        try:
            if self.mode == EvaluationMode.DIRECT_INTERNAL:
                return self.evaluate_direct_internal(statement)
            elif self.mode == EvaluationMode.DIRECT_SEARCH:
                return self.evaluate_direct_search(statement)
            elif self.mode == EvaluationMode.DECOMP_INTERNAL:
                return self.evaluate_decomp_internal(statement)
            elif self.mode == EvaluationMode.DECOMP_SEARCH:
                return self.evaluate_decomp_search(statement)
        except Exception as e:
            print(f"Error during evaluation with mode {self.mode}: {e}")
            return None


def perform_check(
        df: pd.DataFrame,
        mode: EvaluationMode,
        model_name: str = "gpt-4o-mini",
        search_provider: SearchProvider = SearchProvider.DUCKDUCKGO,
        max_workers: int = 1
) -> dict:
    """Process a dataset using the specified evaluation mode."""
    evaluator = FactualityEvaluator(
        mode=mode,
        model_name=model_name,
        search_provider=search_provider,
    )
    predictions = []
    labels = []

    def process_row(row):
        try:
            report, statement = evaluator.evaluate(row['claim'])
            return statement, report, row['label']
        except Exception as e:
            print(f"Error processing row: {e}")
            return None, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            statement, report, label = future.result()
            if report["factuality_score"] is not None:
                if report["factuality_score"] != label:
                    print(statement)
                    print(report["factuality_score"], label)
                    print(report["summary"] + "\n")
                predictions.append(report["factuality_score"])
                labels.append(label)

    # for _, row in df.iterrows():
    #     prediction, label = process_row(row)
    #     predictions.append(prediction)
    #     labels.append(label)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision_score(labels, predictions, average='macro'),
        "macro_recall": recall_score(labels, predictions, average='macro'),
        "macro_f1": f1_score(labels, predictions, average='macro')
    }
    return {
        "predictions": predictions,
        "labels": labels,
        "metrics": metrics,
        "processed_samples": len(predictions),
        "total_samples": len(df)
    }

    # Preprocess


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Configuration
    dataset = "fever"
    split = "train"
    model = "gpt-4o-mini"
    mode = EvaluationMode.DECOMP_INTERNAL

    # Load dataset
    df = pd.read_parquet(f"hf://datasets/ComplexDataLab/Misinfo_Datasets/{dataset}/{dataset}_{split}.parquet")[
        ["claim", "label"]]

    # Preprocess FEVER/FEVEROUS dataset
    if dataset in ["fever", "feverous"]:
        df = df[df['label'].isin(['SUPPORTS', 'REFUTES', 'supports', 'refutes'])]
        df['label'] = df['label'].map({
            'SUPPORTS': 1,
            'supports': 1,
            'REFUTES': 0,
            'refutes': 0
        })

        # Balance dataset
        supports = df[df['label'] == 1].sample(n=750, random_state=42)
        refutes = df[df['label'] == 0].sample(n=750, random_state=42)
        df = pd.concat([supports, refutes]).sample(frac=1, random_state=42).reset_index(drop=True)
    elif dataset == "liar_new":
        df['label'] = df['label'].map({
            'false': 0,
            'pants-fire': 0,
            'barely-true': 0,
            'half-true': 0,
            'mostly-true': 1,
            'true': 1
        })

    # Print initial dataset statistics
    print("\nDataset Statistics:")
    print(df['label'].value_counts())

    print(f"\nRunning evaluation with mode: {mode.value}")

    results = perform_check(
        df=df,
        mode=mode,
        model_name=model,
        search_provider=SearchProvider.DUCKDUCKGO,
        max_workers=1
    )

    # Print metrics
    print("\nPerformance Metrics:")
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value:.3f}")

    data = {'Predictions': results['predictions'], 'Labels': results['labels']}
    df = pd.DataFrame(data)

    # Saving the DataFrame to a CSV file
    df.to_csv(f'{dataset}_{mode.value}_{model}_results.csv', index=False)
