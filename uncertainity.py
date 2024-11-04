"""
Standalone script demonstrating the logic behind
https://arxiv.org/abs/2409.00009 with DuckDuckGo search.
"""
from dotenv import load_dotenv
load_dotenv()

import re
from typing import Any, Dict, List, Optional, NamedTuple
from math import exp
import numpy as np

import openai
from duckduckgo_search import DDGS


MAX_NUM_TURNS: int = 10
MAIN_AGENT_MODEL_NAME: str = "gpt-4o-mini"
MAX_SEARCH_RESULTS: int = 10
TEMPERATURE: float = 0.5

def search(query: str) -> str:
    """
    Invoke web search.

    Params:
        query: str

    Returns:
        Summarized response to search query.
    """
    print(query)
    res = ""
    results = DDGS().text("python programming", max_results=MAX_SEARCH_RESULTS, backend="http")
    results = [r for r in results if "politifact.com" not in r.get("href")][:MAX_SEARCH_RESULTS]
    # for doc in results:
    #     res += f"Title: {doc['title']} Content: {doc['body'][:1600]}\n"
    # response = openai.chat.completions.create(
    #     messages={
    #         "role": "user",
    #         "content": f"Please summarize the searched information for the query. Summarize your findings, taking into account the diversity and accuracy of the search results. Ensure your analysis is thorough and well-organized.\nQuery: {query}\nSearch results: {res}",
    #     }, 
    #     model=MAIN_AGENT_MODEL_NAME,
    #     logprobs=True
    # ).choices[0].message.content
    
    all_logprobs = []
    for doc in results:
        res += f"Title: {doc['title']} Content: {doc['body'][:1600]}\n"
        # Get logprob of this search result being relevant
        search_quality_prompt = f"On a scale of 0-100, rate how relevant and reliable the following information is to evaluating the statement: {statement}\n{res}"
        search_quality_score = openai.chat.completions.create(
            messages={"role": "user", "content": search_quality_prompt},
            model=MAIN_AGENT_MODEL_NAME,
            temperature=TEMPERATURE
        ).choices[0].message.content
        all_logprobs.append(float(search_quality_score))
        print("Search Quality Score:", search_quality_score)


    response = openai.chat.completions.create(
        messages={
            "role": "user",
            "content": f"Please summarize the searched information for the query. Summarize your findings, taking into account the diversity and accuracy of the search results. Ensure your analysis is thorough and well-organized.\nQuery: {query}\nSearch results: {res}",
        }, 
        model=MAIN_AGENT_MODEL_NAME,
        logprobs=True
    ).choices[0].message.content
    print("Summarized Search Response:", response)
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


def _extract_prediction_or_none(assistant_response: str) -> Optional[str]:
    """
    Try to extract "Factuality: 1" (or 0, or 0.5) from main agent response.

    Response:
        Prediction value (as a string) if matched.
        None otherwise.
    """

    match = re.search(r"Factuality:\s(\d+\.?\d?)", assistant_response)
    if match is None:
        return None

    return match.group(1)


def optimize_statement(statement: str) -> str:
    """
    Optimize the claim statement for better search results.

    Returns:
        Optimized claim statement.
    """
    # ask openai model to optimize the statement, to find the missiing time, locationa nd any ambiguity defination, etc.
    # based on the internal knowledge, complete the statemnet and make it more clear and less ambiguous
    
    prompt = """
    Refine the given statement to identify and address missing information, such as time, location, and ambiguous definitions. Use internal knowledge to complete the statement, making it more complete, clear, and unambiguous.

    # Steps

    1. **Analyze the Statement**: Carefully read the provided statement and identify any missing time and location details. Look for any words or phrases that are ambiguous.
    2. **Identify Missing Elements**:
        - Determine if time or location information is missing and needs to be added.
        - Identify specific words or phrases that could be interpreted in multiple ways or are undefined.
    3. **Add Suggested Details**: Based on internal knowledge, add plausible time, location, or definition details to avoid ambiguity.
    4. **Reword for Clarity**: Rewrite the statement to make it more cohesive, logically structured, and free from potential misunderstanding.
    5. **Review for Completeness**: Evaluate the enhanced statement to ensure all previously identified gaps have been filled.

    # Output Format

    - Only provide a refined version of the statement that is complete and has its ambiguities clearly resolved without any additional explanation and prefix and suffix.

    # Example

    **Input**: "The meeting will happen sometime soon with international partners."

    **Output**: 
    "The meeting will take place on September 15th at 10 AM GMT in Geneva, Switzerland, with international partners from Europe, North America, and Asia."

    **Explanation of Changes**: Added specific time ("September 15th at 10 AM GMT") and location ("Geneva, Switzerland") to make the statement clearer and more actionable. Clarified "international partners" by specifying involved regions.
    """
    # - Include 1-2 sentences at the end explaining what was changed or added to make it more clear and complete.

    response = openai.chat.completions.create(
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": statement}],
        model=MAIN_AGENT_MODEL_NAME,
        temperature=TEMPERATURE
    ).choices[0].message.content
    return response


def uncertainty(statement: str, response: List[Dict]) -> float:
    # ask openai model, to generate a 0-100 score, based on the statement
    # and the response
    instructions = [
        "On a scale of 0-100, where 0 is highly uncertain and 100 is fully certain, output your confidence that the information found in the overall result to the search query is accurate and comprehensive. Only give the output.\nStatement: {statement}\nResponse: {response}",
        "On a scale of 0-100, output your confidence that the information found in the overall statement is accurate and comprehensive. Only give the output.\nStatement: {statement}\nResponse: {response}",
        """Based on the following information, output your confidence, on a scale of 0-100 where 0 is highly uncertain and 100 is fully certain,  that the result of the search query made was accurate and comprehensive with regard to the topic queried. Don't write an explanation, only give the number.
The search query was made as one part of a process to evaluate the veracity of this statement: {statement}

The analysis of the search results by the searching agent was: {response}

Note that your score should reflect confidence in the accuracy and comprehensiveness with regard to the topic of the search query. In other words, the extent to which it found the information it sought. It should not reflect whether it comprehensively answered everything about the statement, because there can be other search queries and other analysis taking place too."""
    ]
    scores = []
    for instruction in instructions:
        response = openai.chat.completions.create(
            messages=[{
                "role": "user",
                "content": instruction.format(statement=statement, response=response)
            }],
            model=MAIN_AGENT_MODEL_NAME,
            temperature=TEMPERATURE
        ).choices[0].message.content
        scores.append(float(response))
    print(f"Uncertainty Scores for {len(scores)} instructions:", scores)
    return sum(scores) / len(scores)

if __name__ == "__main__":

    statement = input("Enter statement to verify and press ENTER: ")
    statement = optimize_statement(statement)
    print("Optimized Statement:", statement)

    initial_query = f"""\
    You have access to a search engine tool. To invoke search, \
    begin your query with the phrase "SEARCH: ". You may invoke the search \
    tool as many times as needed. 

    Your task is to analyze the factuality of the given statement.

    Statement: {statement}

    After providing all your analysis steps, summarize your analysis \
    and state "True statement; Factuality: 1" if you think the statement \
    is factual, or "False statement; Factuality: 0" otherwise.
    """

    context: Any = [{"role": "user", "content": initial_query}]


    for turn in range(MAX_NUM_TURNS):
        response = openai.chat.completions.create(
            messages=context, model=MAIN_AGENT_MODEL_NAME, logprobs=True, temperature=TEMPERATURE
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
            search_response, serach_quality = search(search_request_match.matched_content)
            context += [
                {"role": "assistant", "content": search_request_match.content_up_to_match},
                {"role": "user", "content": f"Search result: {search_response} With quality: {serach_quality}"},
            ]
            continue
        else:
            context += [{"role": "assistant", "content": main_agent_message}]


        print(f"Uncertainty Score for Round {turn}:", uncertainty(statement, main_agent_message))

        prediction_match = _extract_prediction_or_none(main_agent_message)
        
        if prediction_match is not None:
            print("=" * 20, "Final Prediction", "=" * 20)
            linear_probs = [np.round(exp(token.logprob) * 100, 2) for token in response.choices[0].logprobs.content]
            text_tokens = [token.token for token in response.choices[0].logprobs.content]

            # extract the logprobs of the prediction
            # for i, token in enumerate(response.choices[0].logprobs.content):
            #     for j, char in enumerate(['F', 'actual', 'ity', ':', ' ']):
            #         if  != char:
            #             break
            #     else:
            #         print("Logprobs of the prediction:", response.choices[0].logprobs.token_logprobs[i-6:i+1])
            #         break
            print(f"Prediction: {prediction_match}")
            for i in range(len(text_tokens)):
                if text_tokens[i+1] == 'actual' and text_tokens[i+2] == 'ity' and text_tokens[i+3] == ':' and text_tokens[i+4] == ' ':
                    print("Logprobs of the prediction:", linear_probs[i+5]) 
                    break

            # print(main_agent_message)
            # print(linear_probs[-10:])
            # print(text_tokens[-10:])
            break