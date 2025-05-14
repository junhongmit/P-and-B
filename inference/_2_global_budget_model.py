import asyncio
import time

from . import *
from utils import *
from utils.logger import *
from utils.prompt_list import *
from utils.utils import *

class GlobalBudget_Model:
    _prompt = {
        "system": textwrap.dedent("""\
        {instruction}

        Please reason step by step, and conclude your answer in the following format:
        
        {output_format}
        """),

        "user": textwrap.dedent("""\
        Question: {query}
        Reference: {reference}
        Let's think step by step and use less than {budget} tokens.
        Output: <think>\n""")
    }

    def __init__(
        self,
        instruction: str = "",
        output_format: str = "",
        use_label_level: bool = True,
        evaluator = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "global_budget"
        self.instruction = instruction
        self.output_format = output_format
        self.use_label_level = use_label_level
        self.evaluator = evaluator
        self.logger = logger

    @llm_retry(max_retries=10, default_output=("", 0))
    async def reasoning(self, query: Query):
        # Given a query and its solving route, ask LLM to solve it by using the given budget.
        system_prompt = self._prompt["system"].format(
            instruction=self.instruction,
            output_format=self.output_format
        )
        user_message = self._prompt["user"].format(
            query=query.description,
            reference=query.reference if query.reference else "None.",
            level=level_map[query.level][0],
            budget=level_map[query.level][1]
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
        
        q = Query(
            description=query,
            reference=reference,
            answer=answer,
            level=level
        )

        if not self.use_label_level:
            level = await evaluate_level(query, self.loader.benchmark_questions, self.logger)
        q.level = level
        q.budget = level_map[level][1]

        output, usage = await self.reasoning(q)

        token_stats = extract_token_stats(usage)
        score, explanation = self.evaluator(query, output, answer)
        self.logger.add_stat({
            "id": id,
            "query": query,
            "answer": answer,
            "prediction": output,
            **token_stats,
            "score": score,
            "explanation": explanation
        })
        self.logger.update_progress({"last_question_total": round(time.time() - start_time, 2)})
