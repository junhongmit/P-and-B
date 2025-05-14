import asyncio
import re
import time

from . import *
from utils import *
from utils.data import *
from utils.logger import *
from utils.prompt_list import *
from utils.utils import *

class PlannedLocalUniform_Model:
    _prompt = {
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
        evaluator = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "planned_local_uniform"
        self.instruction = instruction
        self.evaluator = evaluator
        self.logger = logger

    @llm_retry(max_retries=10, default_output=("", 0))
    async def reasoning(self, query: Query):
        # Displaying the extracted sub-questions and hints
        budget = level_map[query.level][1] // len(query.steps)
        decomposed_str = "\n\n".join([
            f"{step} Please only think a little, and directly solve it using up to {budget} words." 
            for step in query.steps
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

        q = Query(
            description=query,
            steps = steps,
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
            # "budget": query.budget,
            **token_stats,
            "score": score,
            "explanation": explanation
        })
        self.logger.update_progress({"last_question_total": round(time.time() - start_time, 2)})
