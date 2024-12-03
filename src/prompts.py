
class PromptManager:
    @staticmethod
    def get_enrichment_prompt(statement: str) -> str:
        return f"""\
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

    # @staticmethod
    # def get_search_summarization_prompt(statement: str, question: str) -> str:
    #     return f"""\
    #     You are given a user question, and please write clean, concise and accurate answer to the question.\
    #     Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. \
    #     Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat.

    #     If you cannot confidently provide a complete or accurate answer based on your knowledge, \
    #     invoke the search engine tool to find additional information. To invoke the search engine, begin your query with the phrase \
    #     "SEARCH: " followed by the query.

    #     Statement: {statement + ", " + question}
    #     """

    @staticmethod
    def get_factuality_analysis_prompt(statement: str, context: str) -> str:
        return f"""\
        Based on the following gathered information, analyze the factuality of the original statement. \
        Summarize key supporting or contradicting details.

        Original Statement: {statement}
        Contextual Information: {context}

        Return the final output in dictionary:
        {{
            "factuality": "True statement; Factuality: 1" or "False statement; Factuality: 0",
            "summary": "A brief explanation",
            "key_facts": ["Top supporting or contradicting facts"]
        }}
        """
        
    @staticmethod
    def get_search_summarization_prompt(query: str, result: str) -> str:
            return f"You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable. \
            Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say ""information is missing on ..."" followed by the related topic, if the given context do not provide sufficient information. " + f"\nQuery: {query}\nSearch results: {result}"