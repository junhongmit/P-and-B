from datasets import load_dataset
import random
import textwrap
from typing import AsyncGenerator, Any, Dict, List

from utils.data import *
from utils.logger import *
from utils.math import *
from utils.qwen_math_parser import *

class MathDatasetLoader(BaseDatasetLoader):
    _instruction_prompt = "You are a math student being tested on your math problem-solving skills in an exam."
    _output_format = textwrap.dedent("""\
    **Final Answer:** Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.
        
    Where [answer] is the final number or expression *MUST BE* in a valid latex format that solves the problem.
    If you have a list of answers, simply concat them into a comma-sparated list, for example: $\\boxed{{1, \\sqrt{{2}}, 3}}$.
    """)

    def __init__(self, 
                 config: Dict[str, Any],
                 logger: BaseProgressLogger = DefaultProgressLogger(),
                 **kwargs):
        super().__init__(config, **kwargs)
        
        self.logger = logger
        self.config = config
        if self.config.get("decomposed", False):
            with open("dataset/MATH-500/decomposed_questions.json", "r", encoding="utf-8") as f:
                decomposed = json.load(f)
            self.dataset = decomposed['stats']
        else:
            math_dataset = load_dataset("HuggingFaceH4/MATH-500")
            self.dataset = math_dataset['test']
        
        self.data_generator = enumerate(self.dataset)

        # Divide questions into bins for sampling
        self.level_bins = [[] for _ in range(5)]
        for data in self.dataset:
            self.level_bins[data['level'] - 1].append(data)
        self.create_benchmark_questions()

    def create_benchmark_questions(self):
        # Sample one item from each level bin (if available)
        self.benchmark_questions = []
        for i, bin in enumerate(self.level_bins):
            if bin:
                self.benchmark_questions.append(random.choice(bin))
        return self.benchmark_questions
    
    async def load_data(self) -> AsyncGenerator[Dict[str, Any], None]:
        while True:
            try:
                question_id, question = next(self.data_generator)
            except StopIteration:
                break  # Exit the loop when there is no more data.

            question_id = f"{question_id}"
            if question_id in self.logger.processed_questions:
                continue

            # Key remapping
            if not self.config.get("decomposed", False):
                question["query"] = question["problem"]
            
            yield {
                "id": question_id,
                "query": question["query"],
                "steps": question.get("steps", []),
                "decomposed": question.get("decomposed", ""),
                "credits": question.get("credits", []),
                "reference": "",
                "solution": question["solution"],
                "answer": question["answer"],
                "level": question["level"],
            }

    def eval(
        self,
        query: str,
        pred: str,
        ans: str
    ):
        pred = strip_string(extract_answer(pred, 'math'))

        ans = memoized_canonical_form(ans)
        pred = memoized_canonical_form(pred)

        return math_equal(ans, pred), ""
    