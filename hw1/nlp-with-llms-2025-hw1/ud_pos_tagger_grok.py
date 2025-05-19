# See https://docs.x.ai/docs/guides/structured-outputs 
# --- Imports ---
import os
from openai import OpenAI

from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional

# Limit: 5 requests per second 
# Context: 131,072 tokens
# Text input: $0.30 per million
# Text output: $0.50 per million
model = 'grok-3-mini'

# --- Define Pydantic Models for Structured Output ---

# --- Define the Universal Dependencies POS Tagset (17 core tags) as an enum ---
class UDPosTag(str, Enum):
    ADJ = "ADJ"
    ADP = "ADP"
    ADV = "ADV"
    AUX = "AUX"
    CCONJ = "CCONJ"
    DET = "DET"
    INTJ = "INTJ"
    NOUN = "NOUN"
    NUM = "NUM"
    PART = "PART"
    PRON = "PRON"
    PROPN = "PROPN"
    PUNCT = "PUNCT"
    SCONJ = "SCONJ"
    SYM = "SYM"
    VERB = "VERB"
    X = "X"

class TokenPOS(BaseModel):
    token: str = Field(description="The token in the sentence.")
    pos_tag: UDPosTag = Field(description="The POS tag for the token.")

class SentencePOS(BaseModel):
    tokens: List[TokenPOS] = Field(description="A list of tokens in the sentence, each with its corresponding POS tag.")

class TaggedSentences(BaseModel):
    """Represents a list of sentences with their tagged tokens."""
    sentences: List[SentencePOS] = Field(description="A list of sentences, each containing tagged tokens.")

class Explanation(BaseModel):
    explanation: str = Field(description="Explanation of the POS tagging process.")
    category: str = Field(description="Category of the explanation.")

class TokenizedSentence(BaseModel):
    tokens: List[str] = Field(description="A list of tokens in the sentence.")

# --- Configure the Grok API ---
# Get a key https://console.x.ai/team 
# Use os.environ.get for production environments.
# For Colab/AI Studio, you might use userdata.get
# Example:
# from google.colab import userdata
# GROK_API_KEY = userdata.get('GROK_API_KEY')
# genai.configure(api_key=GROK_API_KEY)

# Make sure to replace "YOUR_API_KEY" with your actual key if running locally
# and not using environment variables or userdata.
try:
    # Attempt to get API key from environment variable
    os.environ["GROK_API_KEY"] = "xai-UNG5bAHzI5gncecdW0r1xR5yWZGquSFitOeuawFUtmICUNcKCDpaQGwrfd8xyvfJF8CG7xTW4fuWmx8N"
    api_key = os.environ.get("GROK_API_KEY")
    if not api_key:
        # Fallback or specific instruction for local setup
        # Replace with your actual key if needed, but environment variables are safer
        api_key = "xai-UNG5bAHzI5gncecdW0r1xR5yWZGquSFitOeuawFUtmICUNcKCDpaQGwrfd8xyvfJF8CG7xTW4fuWmx8N"
        if api_key == "YOUR_API_KEY":
           print("⚠️ Warning: API key not found in environment variables. Using placeholder.")
           print("   Please set the GROK_API_KEY environment variable or replace 'YOUR_API_KEY' in the code.")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

except Exception as e:
    print(f"Error configuring API: {e}")
    print("Please ensure you have a valid API key set.")
    # Depending on the environment, you might want to exit here
    # import sys
    # sys.exit(1)


# --- Function to Perform POS Tagging ---

def tag_sentences_ud(text_to_tag: str) -> Optional[TaggedSentences]:
    """
    Performs POS tagging on the input list of sentences using the Grok API and
    returns the result structured according to the TaggedSentences Pydantic model.

    Args:
        text_to_tag: The sentence or text to be tagged.

    Returns:
        A SentencePOS object containing the tagged tokens, or None if an error occurs.
    """
    # Construct the prompt
    prompt = f"""Tag the following text with POS tags:\n{text_to_tag}\n\n"""
    completion = client.beta.chat.completions.parse(
        model="grok-3",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text_to_tag},
        ],
        response_format=TaggedSentences,
    )
    # print(completion)
    res = completion.choices[0].message.parsed
    return res

# --- Function to Get Tag Explanation ---

def get_tag_explanation(prompt: str) -> str:
    completion = client.beta.chat.completions.parse(
        model="grok-3",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": prompt},
        ],
        response_format=Explanation,
    )
    res = completion.choices[0].message.parsed
    return res.explanation, res.category


def tokenize_sentence(instruction: str, prompt: str) -> str:
    """
    given an original sentence, it must return a tokenized list of tokens
    according to the CoNLL segmentation guidelines in a JSON format.
    """
    completion = client.beta.chat.completions.parse(
        model="grok-3",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
        response_format=TokenizedSentence,
    )
    res = completion.choices[0].message.parsed
    return res.tokens

# --- Example Usage ---
if __name__ == "__main__":
    # example_text = "The quick brown fox jumps over the lazy dog."
    example_text = """
What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?
Google Search is a web search engine developed by Google LLC.
It does n't change the company 's intrinsic worth , and as the article notes , the company might be added to a major index once the shares get more liquid .
I 've been looking at the bose sound dock 10 i ve currently got a jvc mini hifi system , i was wondering what would be a good set of speakers .
which is the best burger chain in the chicago metro area like for example burger king portillo s white castle which one do you like the best ?
"""
    # example_text = "החתול המהיר קופץ מעל הכלב העצלן." # Example in Hebrew

    print(f"\nTagging text: \"{example_text}\"")

    tagged_result = tag_sentences_ud(example_text)

    if tagged_result:
        print("\n--- Tagging Results ---")
        for s in tagged_result.sentences:
            for token_pos in s.tokens:
                token = token_pos.token
                tag = token_pos.pos_tag
                # Handle potential None for pos_tag if model couldn't assign one
                ctag = tag if tag is not None else "UNKNOWN"
                print(f"Token: {token:<15} {str(ctag)}")
                print("----------------------")
    else:
        print("\nFailed to get POS tagging results.")