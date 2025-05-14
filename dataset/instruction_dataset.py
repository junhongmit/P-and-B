from datasets import load_dataset
from evaluate import load
import random
import textwrap
from typing import AsyncGenerator, Any, Dict, List

from utils.data import *
from utils.logger import *
from utils.qwen_math_parser import *

rouge_metric = load('rouge')
def rouge(prediction, ground_truth):
    score = rouge_metric.compute(
        predictions=[prediction],
        references=[ground_truth],
        use_stemmer=True,
        rouge_types=['rougeL']
    )
    return float(score['rougeL'])

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

class InstructionDatasetLoader(BaseDatasetLoader):
    _instruction_prompt = "You are a student being tested on your problem-solving skills in an exam."
    _output_format = textwrap.dedent("""\
    **Final Answer:** Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.
        
    Where [answer] is just the final number or expression in latex format that solves the problem.
    If you have a list of answers, simply concat them into a comma-sparated list, for example: $\\boxed{{1, \\sqrt{{2}}, 3}}$.
    """)

    def __init__(
        self, 
        config: Dict[str, Any],
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        self.logger = logger

        if self.config.get("decomposed", False):
            with open("dataset/NaturalInstruction-Sampled-500/decomposed_questions.json", "r", encoding="utf-8") as f:
                decomposed = json.load(f)
        else:
            with open("dataset/NaturalInstruction-Sampled-500/sampled_500_natural_instruction.json", "r", encoding="utf-8") as f:
                decomposed = json.load(f)
        self.dataset = decomposed['stats']
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
            
            yield {
                "id": question_id,
                "query": question["query"],
                "steps": question.get("steps", []),
                "decomposed": question.get("decomposed", ""),
                "credits": question.get("credits", []),
                "reference": "",
                "solution": "",
                "answer": question["answer"],
                "level": question["level"],
            }

    def eval(
        self,
        query: str,
        pred: str,
        ans: str
    ):
        pred = strip_string(extract_answer(pred, 'instruction'))
        return rouge(pred, ans), ""