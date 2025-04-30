from openai import OpenAI
import time
import os
import pickle
import uuid

query_prompt = """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Français, Español, Deutsch, or 日本語 should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {query}

Search Results:
{source_text}
"""

# Initialize the OpenAI client with modern syntax
openai_client = OpenAI(api_key=os.getenv("VENICE_API_KEY"), base_url="https://api.venice.ai/api/v1")


def generate_answer(
    query, sources, num_completions=1, temperature=0.5, verbose=False, model="llama-3.2-3b"
):
    source_text = "\n\n".join(
        [f"### Source {idx + 1}:\n{source}\n\n\n" for idx, source in enumerate(sources)]
    )
    prompt = query_prompt.format(query=query, source_text=source_text)

    while True:
        try:
            if verbose:
                print("Calling OpenAI Client...")

            response = openai_client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=1024,
                top_p=1,
                n=num_completions,
                messages=[{"role": "user", "content": prompt}],
            )

            if verbose:
                print("Response received.")

            # Ensure the output directory exists
            output_dir = "response_usages_16k"
            os.makedirs(output_dir, exist_ok=True)

            # Save token usage to disk
            filename = os.path.join(output_dir, f"{uuid.uuid4()}.pkl")
            with open(filename, "wb") as f:
                pickle.dump(response.usage, f)

            return [choice.message.content + "\n" for choice in response.choices]

        except Exception as e:
            print("Error in calling OpenAI API:", e)
            time.sleep(15)
            continue
