import asyncio
from dataclasses import dataclass, field
import functools
import json
import openai
import os
from typing import Any, Dict, List, Optional

from . import *
from utils.logger import *
from utils.prompt_list import *

@dataclass
class Query:
    description: str                    # Question
    reference: Optional[str] = None     # Accompany references
    steps: Optional[List[str]] = None   # Decomposed subquestions
    credits: Optional[List[int]] = None # Credit of each subquestion
    answer: Optional[str] = None        # Groundtruth answer
    level: Optional[int] = None         # Difficulty level
    budget: Optional[int] = None        # Token budget
    
@dataclass
class Route:
    query: Query
    subqueries: Optional[List[Query]] = None

@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

level_map = {
    1: ("simple", 200),
    2: ("simple", 250),
    3: ("medium", 350),
    4: ("medium", 450),
    5: ("hard", 600)
}

def llm_retry(max_retries=5, default_output=None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            self = args[0] if args else None
            logger = getattr(self, 'logger', getattr(kwargs, 'logger', DefaultProgressLogger()))
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except openai.APIConnectionError as e:
                    if logger: logger.error(f"[Retry {attempt+1}/{max_retries}] API connection failed", exc_info=True)
                    await asyncio.sleep(min(2 ** attempt, 30))  # Exponential backoff (2s, 4s, 8s, etc.)
                except openai.RateLimitError as e:
                    if logger: logger.error(f"[Retry {attempt+1}/{max_retries}] Rate limit hitted", exc_info=True)
                    await asyncio.sleep(min(30 + 2 ** attempt, 90))  # Exponential backoff (2s, 4s, 8s, etc.)
                except json.decoder.JSONDecodeError:
                    if logger: logger.error(f"[Retry {attempt+1}/{max_retries}] JSON Decode error", exc_info=True)
                    await asyncio.sleep(min(2 ** attempt, 30))
                except TypeError:
                    if logger: logger.error(f"[Retry {attempt+1}/{max_retries}] JSON format error", exc_info=True)
                    await asyncio.sleep(min(2 ** attempt, 30))
                except Exception:
                    if logger: logger.error(f"[Retry {attempt+1}/{max_retries}] Unexpected error", exc_info=True)
                    await asyncio.sleep(min(2 ** attempt, 30))
            return default_output
        return wrapper
    return decorator

# We maintain a singleton LLM driver and KG driver
_client = openai.AsyncOpenAI(
    base_url=API_BASE,
    api_key=API_KEY,
    default_headers={'RITS_API_KEY': os.environ["RITS_API_KEY"]} if os.environ.get("RITS_API_KEY") else None
)
logger.info(f"Using {MODEL_NAME} for LLM response.")

@llm_retry(max_retries=5, default_output="")
async def complete_response(
    prompt, 
    max_tokens=8192, 
    temperature=0.1, 
    top_p=0.9, 
    timeout=3600, 
    logger: BaseProgressLogger = DefaultProgressLogger(),
    return_raw: bool = False,
    **kwargs
) -> str:
    """Asynchronous function to perform text completion using OpenAI's completions API."""
    response = await _client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        **kwargs
    )
    if return_raw:
        return response
    else:
        return response.choices[0].text  # Extract the generated text from the response

@llm_retry(max_retries=5, default_output="")
async def generate_response(
    prompt, 
    max_tokens=8192, 
    temperature=0.1, 
    top_p=0.9, 
    timeout=3600, 
    logger: BaseProgressLogger = DefaultProgressLogger(),
    return_raw: bool = False,
    **kwargs
) -> str:
    """Asynchronous function to evaluate a single answer."""
    response = await _client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        **kwargs
    )
    if return_raw:
        return response
    else:
        return response.choices[0].message.content  # Extract response text

@llm_retry(max_retries=5, default_output="")
async def generate_reasoning_response(
    prompt,
    reasoning_effort="low",  # can be "low", "medium", "high"
    timeout=3600,
    logger: BaseProgressLogger = DefaultProgressLogger(),
    return_raw: bool = False,
    **kwargs
) -> str:
    """
    Asynchronous function to evaluate a single answer using OpenAI's Responses API
    with reasoning models (e.g., o4-mini, o3, o1).
    """
    response = await _client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        reasoning={"effort": reasoning_effort},
        timeout=timeout,
        **kwargs
    )

    if return_raw:
        return response
    else:
        return response.output_text

# Attempt for vLLM based reasoning model
# @llm_retry(max_retries=5, default_output="")
# async def generate_reasoned_response(prompt, 
#                             max_tokens=8192, 
#                             temperature=None, 
#                             top_p=None, 
#                             timeout=3600, 
#                             logger: BaseProgressLogger = DefaultProgressLogger(),
#                             return_raw: bool = False,
#                             **kwargs) -> str:
#     """Asynchronous function to evaluate a single answer."""
#     response = await _client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=prompt,
#         max_tokens=max_tokens,
#         temperature=temperature,
#         top_p=top_p,
#         timeout=timeout,
#         **kwargs
#     )
#     if return_raw:
#         return response
#     else:
#         return {
#             "reasoning_content": response.choices[0].message.reasoning_content,
#             "content": response.choices[0].message.content
#         }

def extract_token_stats(usage) -> dict:
    """
    Normalize token usage for chat/completion/responses APIs.

    Handles:
    - Chat/Completion API: prompt_tokens, completion_tokens, total_tokens
    - Responses API: input_tokens, output_tokens, total_tokens,
                     output_tokens_details.reasoning_tokens

    Returns:
        A dict with keys:
            - prompt_tokens
            - reasoning_tokens (None if not present)
            - completion_tokens
            - total_tokens
    """
    # For chat/completion
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)

    # For responses API
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    output_details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = getattr(output_details, "reasoning_tokens", None) if output_details else None

    return {
        "prompt_tokens": prompt_tokens or input_tokens,
        "reasoning_tokens": reasoning_tokens,
        "completion_tokens": completion_tokens or output_tokens,
        "total_tokens": getattr(usage, "total_tokens", None),
    }

@llm_retry(max_retries=10, default_output=[])
async def evaluate_level(query: Query,
                         benchmark_questions:List,
                         logger: BaseProgressLogger = DefaultProgressLogger()) -> int:
    # Create the benchmark example text
    benchmark_examples = "\n".join(
        f"\nLevel {q['level']} Example:\nQ: {q['problem']}\nA: {q['solution']}\n"
        for q in benchmark_questions
    )
    
    # Ask LLM to predict the question level and return a Query object
    system_prompt = PROMPTS["evaluate_question_level"]["system"].format(
        benchmarks=benchmark_examples
    )
    user_message = PROMPTS["evaluate_question_level"]["user"].format(
        query=query.description
    )
    formatted_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    logger.debug(system_prompt + "\n" + user_message)
    
    # Run all requests asynchronously
    response = await generate_response(
        formatted_prompt,
        max_tokens=1024,
        response_format={"type": "json_object"},
        # extra_body={"guided_json": output_schema}
        logger=logger
    )
    logger.debug(response)
    
    result = maybe_load_json(response).get("evaluated_level", 5)
    return result

def extract_json_objects(text, decoder=json.JSONDecoder()):
    """Find JSON objects in text, and yield the decoded JSON data
    """
    pos = 0
    results = []
    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            results.append(result)
            pos = match + index
        except ValueError:
            pos = match + 1
    return results

def maybe_load_json(text: str, force_load = True) -> object:
    try:
        res = json.loads(text)
    except:
        if force_load:
            res = extract_json_objects(text)
            res = res[0] if len(res) else res
        else:
            return None
    return res

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop
