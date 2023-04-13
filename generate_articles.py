import os
import openai
from pydantic import BaseModel, validator
from tenacity import retry, stop_after_attempt, wait_fixed

class KeywordList(BaseModel):
    keywords: list[str]

    @validator("keywords", pre=True, each_item=True)
    def validate_keywords(cls, keyword):
        if not keyword.strip():
            raise ValueError("Keyword cannot be empty or whitespace.")
        return keyword.strip()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def fetch_articles(api_key, keywords, model="text-davinci-002"):
    openai.api_key = api_key

    prompt = f"Generate articles based on the following keywords: {', '.join(keywords)}"

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )

    article = response.choices[0].text.strip()
    return article

if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise Exception("Please set the OPENAI_API_KEY environment variable")

    keywords = ["artificial intelligence", "machine learning", "neural networks"]
    try:
        keyword_list = KeywordList(keywords=keywords)
        article = fetch_articles(api_key, keyword_list.keywords)
        print(article)
    except ValueError as e:
        print(f"Invalid input: {e}")
    except openai.error.OpenAIError as e:
        print(f"OpenAIError: {e}")
    except Exception as e:
        print(f"Error: {e}")
