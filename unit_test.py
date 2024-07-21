from query import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

MODEL = "llama3"

def test_hkma_fintech():
    assert query_and_validate(
        question = "Elaborate  HK fintech roadmap",
        expected_response = """

        Based on the provided context, it appears that the Hong Kong Fintech Promotion Roadmap (HKMA's future training sessions) aims to:

        1. Facilitate expert-led, practical, and hands-on training in various Fintech verticals and technology overlays.
        2. Bridge the gap between theoretical and academic certification programs by providing more applied knowledge and skills for implementing and using Fintech solutions.

        Additionally, the roadmap may involve:

        1. Establishing a robust framework for promoting Fintech adoption and development in Hong Kong.
        2. Encouraging the use of recent updates to the G20/OECD High-Level Principles on Financial Consumer Protection to ensure consistency with international banking consumer protection practices.
        3. Reviewing and updating relevant regulations, such as the Banking Ordinance, to support the growth and innovation of Fintech in Hong Kong.

        """
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response = expected_response, actual_response = response_text
    )

    model = Ollama(model = MODEL)
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )