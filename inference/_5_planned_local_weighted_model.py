import asyncio
import re
import time
import numpy as np

from . import *
from utils import *
from utils.data import *
from utils.logger import *
from utils.prompt_list import *
from utils.utils import *

def allocate_tokens(credits, total_tokens, schedule_fn):
    """
    credits: list of positive floats summing to 1 (normalized sub-question credits)
    total_tokens: int, total token budget
    schedule_fn: function taking step-index i (0-based) and N, returning a non‑negative weight
    """
    N = len(credits)
    sched_weights = np.array([schedule_fn(i, N) for i in range(N)], dtype=float)
    combined = np.array(credits) * sched_weights
    combined /= combined.sum()
    tokens = np.floor(combined * total_tokens).astype(int)
    diff = total_tokens - tokens.sum()
    for i in np.argsort(-combined)[:diff]:
        tokens[i] += 1
    return tokens

def constant_schedule(i, N):
    return 1.0

def linear_decay(i, N):
    # weight ∝ (N−i)
    return (N - i)

def polynomial_decay(i, N, power=2):
    # weight ∝ (N−i)^power
    return (N - i)**power

def cosine_anneal(i, N):
    # weight ∝ 0.5*(1 + cos(pi * i/(N−1)))
    if N > 1:
        return 0.5*(1 + np.cos(np.pi * i / (N-1))) + 0.1
    else:
        return 1

def exponential_decay(i, N, base=0.9):
    # weight ∝ base^i
    return base**i

class PlannedLocalWeighted_Model:
    _prompt = _prompt = {
        # 1. For simple problems (level 2 or fewer): Only think a little. Provide a concise solution with minimal explanation.
        # 2. For complex problems (level 3 or more): You may think following the sub-questions or feel free to use other methods that works the best towards getting the final answer.  
        "system": textwrap.dedent("""\
        {instruction}
        
        The problem is given by an overall description, difficulty level out of 5, followed by a series of sub-questions as a hint.
        All the credit is given when you provide a correct final answer for the overall problem.
        Please solve the question efficiently and clearly to achieve as much credit as possible."""),

        "user": textwrap.dedent("""\
        Let's start the exam. You are being given this math problem:
        **Problem (100pt):** {question}
        **Reference:** {reference}
        **Level:** {level} out of 5

        You may think following these sub-questions or feel free to use other methods that works the best towards getting the final answer:
        {decomposed}

        Please provide your final answer strictly following the format:
        {output_format}
        
        Output: <think>\n""")
    }

    def __init__(
        self,
        instruction: str = "",
        decay: str = "constant",
        use_label_level: bool = True,
        evaluator = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "planned_local_weighted"
        self.instruction = instruction
        self.decay = decay
        self.use_label_level = use_label_level
        self.evaluator = evaluator
        self.logger = logger

    @llm_retry(max_retries=10, default_output=[])
    async def evaluate_level_batch(self, route: Route, benchmark_questions:List) -> Route:
        # Create the benchmark example text
        benchmark_title = """
        You are an expert question analyst. Your task is to evaluate the difficulty level of a given question by comparing it against a set of benchmark questions of known levels.

        Each level reflects increasing complexity from 1 (easiest) to 5 (most challenging). Evaluate based on the conceptual depth, steps involved in solving, required knowledge, and potential for misdirection.

        Use the following benchmark examples as references:
        """
        
        benchmark_examples = benchmark_title + "\n".join(
            f"\nLevel {q['level']} Example:\nQ: {q['problem']}\nA: {q['solution']}\n"
            for q in benchmark_questions
        )
        
        # Ask LLM to predict the question level and return a Query object
        system_prompt = PROMPTS["evaluate_question_level_batch"]["system"].format(
            benchmarks=benchmark_examples
        )
        user_message = PROMPTS["evaluate_question_level_batch"]["user"].format(
            query=route.query.description,
            route=[subquery.description for subquery in route.subqueries],
        )
        formatted_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        self.logger.debug(system_prompt + "\n" + user_message)
        
        # Run all requests asynchronously
        response = await generate_response(
            formatted_prompt,
            max_tokens=2048,
            response_format={"type": "json_object"},
            # extra_body={"guided_json": output_schema}
            logger=logger
        )
        self.logger.debug(response)
        
        result = maybe_load_json(response)
        if not self.use_label_level:
            route.query.level = result['question']['evaluated_level']
        route.query.budget = level_map[route.query.level][1]
        level_sum = 0
        for idx, subquery in enumerate(route.subqueries):
            subquery.level = result[f'step_{idx + 1}']['evaluated_level']
            level_sum += subquery.level
        for idx, subquery in enumerate(route.subqueries):  
            subquery.budget = route.query.budget * subquery.level // level_sum
            
        return route

    @llm_retry(max_retries=10, default_output=("", 0))
    async def reasoning(self, query: Query):
        decay_func = None
        if self.decay.lower() == "constant":
            decay_func = constant_schedule
        elif self.decay.lower() == "linear":
            decay_func = linear_decay
        elif self.decay.lower() == "polynomial":
            decay_func = polynomial_decay
        elif self.decay.lower() == "exponential":
            decay_func = exponential_decay
        elif self.decay.lower() == "cosine":
            decay_func = cosine_anneal
        
        tokens_per_part = allocate_tokens(
            query.credits, 
            level_map[query.level][1],
            decay_func
        )
        decomposed_str = "\n\n".join([
            f"{step} Please only think a little, and directly solve it using up to {budget} words." 
            for step, budget in zip(query.steps, tokens_per_part)
        ])
            
        # Given a query and its solving route, ask LLM to solve it by using the given budget.
        system_prompt = self._prompt["system"].format(
            instruction=self.instruction
        )
        user_message = self._prompt["user"].format(
            question=query.description,
            reference=query.reference if query.reference else "None.",
            level=query.level,
            decomposed=decomposed_str,
            output_format=self.output_format
        )
        
        output = ""
        if MODEL_NAME in COMPLETION_LLM:
            response = await complete_response(
                system_prompt + "\n" + user_message, 
                return_raw=True
            )
            output = response.choices[0].text
        elif MODEL_NAME in CHAT_LLM:
            response = await generate_response(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                logger=logger,
                return_raw=True
            )
            output = response.choices[0].message.content
        elif MODEL_NAME in REASONING_LLM:
            response = await generate_reasoning_response(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                return_raw=True
            )
            output = response.output_text
        else:
            raise NotImplementedError("This model is not supported")

        self.logger.debug(system_prompt + "\n" + user_message + "\n" + output)
        
        return output, response.usage

    async def process_question(
        self, 
        id: str,
        query: str,
        steps: List[str],
        credits: List[int],
        reference: str = "",
        solution: str = "",
        answer: str = "",
        level: int = None,
        logger=DefaultProgressLogger(),
        **kwargs
    ):
        start_time = time.time()
        
        # route = await self.break_down_question(query)
        # route = await self.evaluate_level_batch(route, self.loader.benchmark_questions)

        q = Query(
            description=query,
            steps=steps,
            credits=credits,
            reference=reference,
            answer=answer,
            level=level
        )

        output, usage = await self.reasoning(q)

        token_stats = extract_token_stats(usage)
        score, explanation = self.evaluator(query, output, answer)
        self.logger.add_stat({
            "id": id,
            "query": query,
            "answer": answer,
            "prediction": output,
            "level": level,
            # "predicted_level": level,
            # "subquery_levels": [subquery.level for subquery in route.subqueries],
            # "budget": query.budget,
            **token_stats,
            "score": score,
            "explanation": explanation
        })
        self.logger.update_progress({"last_question_total": round(time.time() - start_time, 2)})
