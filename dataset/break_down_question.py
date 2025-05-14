import argparse
import time
import textwrap

from dataset.math_dataset import *
from dataset.instruction_dataset import *
from dataset.travelplanner_dataset import *
from utils import *
from utils.logger import *
from utils.utils import *
from utils.math import *
from utils.qwen_math_parser import *


PROMPTS = {}

PROMPTS["break_down"] = {
    "system": textwrap.dedent("""\
        -Goal-
        You are an experienced expert in {domain} and exam question designer. Your role is to help students break down challenging math problems into a series of simpler, high-level sub-questions.
        We don't want too many detailed sub-questions, which are not beneficial for testing students' ability in an exam. Each sub-question should build on the previous one so that, once all have been answered, the complete solution is clear.
        Your output should be a list of sub-questions with brief hints explaining the purpose of each step, but you should not reveal your internal chain-of-thought either the final solution.

        Instructions for Decomposition:
        First, analyze the problem and identify the key ideas needed to solve it. Then, generate a series of 2 to 5 sub-questions that lead the student step by step to the complete solution.
        The difficulty level of the problem is presented out of 5, wher 1 is easy, and 5 is hard. Please adjust the number of sub-questions based on the level. Ideally, we want fewer sub-questions for
        easy problems and more sub-questions for challenging problems.
        DO NOT perform reasoning, directly output those sub-questions based on your gut feelings; only output the list of sub-questions with brief hints for each.
        Your answer should be a list of numbered sub-questions. Each sub-question should have a brief accompanying hint that explains what the student will achieve by answering that part. 

        Example Decomposition:
        **Problem:** Find the remainder when \\(9 \\times 99 \\times 999 \\times \\cdots \\times \\underbrace{{99\\cdots9}}_{{\\text{{999 9's}}}}\\) is divided by 1000.
        **Level:** 3 out of 5

        **Decomposed Sub-questions:**

        1. Compute the product modulo 8.
        Hint: Simplify each term using \\(10 \\equiv 2 \\mod 8\\), noting that \\(10^k \\equiv 0 \\mod 8\\) for \\(k \\geq 3\\), leading to terms of \\(-1 \\mod 8\\).

        2. Compute the product modulo 125.
        Hint: Recognize \\(10^3 \\equiv 0 \\mod 125\\), so terms for \\(k \\geq 3\\) become \\(-1 \\mod 125\\). Calculate the product of the first two terms and combine with the remaining terms.

        3. Solve the system of congruences using the Chinese Remainder Theorem.
        Hint: Combine the results from modulo 8 and modulo 125 to find a common solution modulo 1000.
    """),

    "user": textwrap.dedent("""\
        A student has presented you with the following math problem:
        Problem: {problem}
        Level: {level} out of 5
        **REMEMBER**, you are not allowed to think about it, please directly generate the answer in the following:
        Decomposed Sub-questions:
    """)
}

PROMPTS["evaluate_question_level_batch"] = {
    "system": textwrap.dedent("""\
        You are an experienced expert in {domain} and exam question designer. Your task is to evaluate the difficulty level of a given exam problem and its sub-questions by comparing it against a set of benchmark questions of known levels.
        Based on their levels, you will need to assign each subquestion a portion of the credits (assuming the total credit points is 100 for the whole problem).

        Each level reflects increasing complexity from 1 (easiest) to 5 (most challenging). Evaluate based on the conceptual depth, steps involved in solving, required knowledge, and potential for misdirection.

        Use the following benchmark examples as references:
        
        {benchmarks}

        1. You will be provided a question and its subquestions. You will evaluate the difficulty level of the problem and its sub-questions.
        Assuming the whole problem is worth 100 points, you assign each sub-question a portion of the score points.
        - Adhere to the given subquestions, and DO NOT make new subquestions.
        - Sum of each subquestion's credits MUST EQUAL to 100.
        
        2. You must return the result in a structured JSON format:
        {{
        "problem": {{"reason": "...", "evaluated_level": level_q}}
        "1": {{"reason": "...", "evaluated_level": level_1, "credit": credit_1}},
        "2": {{"reason": "...", "evaluated_level": level_2, "credit": credit_2}},
        ...}}
        where
        - "reason": a short explanation (up to 50 words) of your level assessment.
        - "evaluated_level": an integer from 1 to 5 indicating your judgment.
        - "credit": an integer between 1 to 100 indicating when the question is solved correctly, how many credit can be given.
    """),

    "user": textwrap.dedent("""\
        Evaluate the level of the following question:
        Problem: {problem}
        Sub-questions: {steps}
        Output:""")
}

async def break_down_question(
    id: str,
    query: str,
    reference: str = "",
    solution: str = "",
    answer: str = "",
    level: int = None,
    domain: str = "",
    logger=DefaultProgressLogger(),
    **kwargs
) -> List[str]:
    start_time = time.time()
    
    system_prompt = PROMPTS["break_down"]["system"].format(
        domain=domain
    )
    user_message = PROMPTS["break_down"]["user"].format(
        problem=query,
        level=level
    )

    response = await generate_response(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )
    logger.info(user_message + "\n" + response)
    
    pattern = r"(\d+\..*?)(?=\nHint:)(?:\nHint:\s*(.*?))(?:\n|$)"
    matches = re.findall(pattern, response, re.DOTALL)

    if len(matches) == 0:
        matches = [("1. Directly solve the problem.", "None.")]
    steps = [f"{match[0]} Hint: {match[1].strip()}" for match in matches]

    logger.add_stat({
        "id": id,
        "query": query,
        "steps": steps,
        "reference": reference,
        "solution": solution,
        "answer": answer,
        "level": level,
    })
    logger.update_progress({"last_question_total": round(time.time() - start_time, 2)})

async def assess_question(
    id: str,
    query: str,
    steps: List[str],
    reference: str = "",
    solution: str = "",
    answer: str = "",
    level: int = None,
    domain: str = "",
    logger=DefaultProgressLogger(),
    **kwargs
):
    start_time = time.time()

    # Ask LLM to predict the question level and return a Query object
    system_prompt = PROMPTS["evaluate_question_level_batch"]["system"].format(
        domain=domain,
        benchmarks=benchmark_examples
    )
    user_message = PROMPTS["evaluate_question_level_batch"]["user"].format(
        problem=query,
        steps=steps,
    )
    response = await generate_response(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"}
    )
    logger.info(user_message + "\n" + response)
    
    result = maybe_load_json(response)

    logger.add_stat({
        "id": id,
        "query": query,
        "steps": steps,
        "credits": [result[f"{idx + 1}"]['credit'] for idx in range(len(result) - 1)],
        "solution": solution,
        "answer": answer,
        "level": level,
        "reference": reference
    })
    logger.update_progress({"last_question_total": round(time.time() - start_time, 2)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Evaluation dataset")
    parser.add_argument("--num-workers", type=int, default=512, help="Number of workers generating the answers")
    parser.add_argument("--queue-size", type=int, default=512, help="Queue size of data loading")
    args = parser.parse_args()

    config = {
        "num_workers": args.num_workers,
        "queue_size": args.queue_size,
        "decomposed": False
    }
    progress_path = f"results/decomposed_questions.json"
    logger = QAProgressLogger(progress_path=progress_path)

    loader = None
    if args.dataset.lower() == 'math':
        domain = "math"
        loader = MathDatasetLoader(
            config=config, 
            logger=logger,
            processor=functools.partial(break_down_question, domain=domain, logger=logger)
        )
        benchmark_examples = "\n".join(
            f"\nLevel {q['level']} Example:\nQ: {q['problem']}\nA: {q['solution']}\n"
            for q in loader.benchmark_questions
        )
    elif args.dataset.lower() == 'instruction':
        domain = "task completions"
        loader = InstructionDatasetLoader(
            config=config, 
            logger=logger,
            processor=functools.partial(break_down_question, domain=domain, logger=logger)
        )
        benchmark_examples = "None"
    elif args.dataset.lower() == 'travelplanner':
        domain = "travel planning"
        loader = TravelPlannerDatasetLoader(
            config=config, 
            logger=logger,
            processor=functools.partial(break_down_question, domain=domain, logger=logger)
        )
        benchmark_examples = "None"
    else:
        raise NotImplementedError("Dataset is not supported ❌")

    loop = always_get_an_event_loop()
    loop.run_until_complete(
        loader.run()
    )

    logger.info(f"Done question decomposition on {args.dataset} ✅")



    with open(progress_path, "r", encoding="utf-8") as f:
        decomposed = json.load(f)
    decomposed["stats"] = sorted(decomposed["stats"], key=lambda x: int(x["id"]))

    logger.processed = set()
    logger.progress_data['stats'] = []
    loader.config["decomposed"] = True
    loader.dataset = decomposed['stats']
    loader.data_generator = enumerate(loader.dataset)
    loader.processor = functools.partial(assess_question, logger=logger)

    loop.run_until_complete(
        loader.run()
    )

    logger.info(f"Done subquestion difficulty evaluation on {args.dataset} ✅")
    logger.info(f"Results stored at {progress_path} ✅")