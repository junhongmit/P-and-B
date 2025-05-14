from datasets import load_dataset
import random
import textwrap
from typing import AsyncGenerator, Any, Dict, List

from utils.data import *
from utils.logger import *
from utils.math import *
from utils.qwen_math_parser import *
from utils.utils import *

level_mapper = {
    "easy": 1,
    "medium": 3,
    "hard": 5
}

class TravelPlannerDatasetLoader(BaseDatasetLoader):
    _instruction_prompt = textwrap.dedent("""\
    You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names.
    Note that all the information in your plan should be derived from the provided data.
    """)
    _output_format = textwrap.dedent("""\
    You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary.
    For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B).

    ***** Example *****
    Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
    Travel Plan:
    Day 1:
    Current City: from Ithaca to Charlotte
    Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
    Breakfast: Nagaland's Kitchen, Charlotte
    Attraction: The Charlotte Museum of History, Charlotte
    Lunch: Cafe Maple Street, Charlotte
    Dinner: Bombay Vada Pav, Charlotte
    Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

    Day 2:
    Current City: Charlotte
    Transportation: -
    Breakfast: Olive Tree Cafe, Charlotte
    Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
    Lunch: Birbal Ji Dhaba, Charlotte
    Dinner: Pind Balluchi, Charlotte
    Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

    Day 3:
    Current City: from Charlotte to Ithaca
    Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
    Breakfast: Subway, Charlotte
    Attraction: Books Monument, Charlotte.
    Lunch: Olive Tree Cafe, Charlotte
    Dinner: Kylin Skybar, Charlotte
    Accommodation: -

    ***** Example Ends *****""")

    

    def __init__(self, 
                 config: Dict[str, Any],
                 logger: BaseProgressLogger = DefaultProgressLogger(),
                 **kwargs):
        super().__init__(config, **kwargs)
        
        self.logger = logger
        travel_dataset = load_dataset('osunlp/TravelPlanner','validation')['validation']
        self.travel_data_dict = {item['query']: item for item in travel_dataset}
        if config.get("decomposed", False):
            with open("dataset/TravelPlanner/decomposed_questions.json", "r", encoding="utf-8") as f:
                decomposed = json.load(f)
            self.dataset = decomposed['stats']
        else:
            self.dataset = travel_dataset
        self.data_generator = enumerate(self.dataset)

        # Divide questions into bins for sampling
        self.level_bins = [[] for _ in range(5)]

        for data in self.dataset:
            if not self.config.get("decomposed", False):
                self.level_bins[level_mapper[data['level']] - 1].append(data)
            else:
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
            if not self.config.get("decomposed", False) and question["reference_information"]:
                question["reference"] = question["reference_information"]
            if not self.config.get("decomposed", False) and isinstance(question["level"], str):
                question["level"] = level_mapper[question['level']]
            
            yield {
                "id": question_id,
                "query": question["query"],
                "steps": question.get("steps", []),
                "decomposed": question.get("decomposed", ""),
                "credits": question.get("credits", []),
                "reference": question["reference"],
                "solution": question.get("solution", ""),
                "answer": question.get("answer", ""),
                "level": question["level"],
            }

    def eval(
        self,
        query: str,
        pred: str,
        ans: str
    ):
        # Need to evaluate using LLM, we defer it later
        return -1, ""

    async def post_eval(
        self,
        query: str,
        pred: str,
        ans: str
    ):
        from dataset.TravelPlanner.eval import eval_score
        # Convert natural language plan to JSON first
        system_prompt = """\
        Please assist me in extracting valid information from a given natural language text and reconstructing it in JSON format, as demonstrated in the following example.
        If transportation details indicate a journey from one city to another (e.g., from A to B), the 'current_city' should be updated to the destination city (in this case, B).
        Use a ';' to separate different attractions, with each attraction formatted as 'Name, City'.
        If there's information about transportation, ensure that the 'current_city' aligns with the destination mentioned in the transportation details (i.e., the current city should follow the format 'from A to B').
        Also, ensure that all flight numbers and costs are followed by a colon (i.e., 'Flight Number:' and 'Cost:'), consistent with the provided example.
        Each item should include ['day', 'current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation'].
        Replace non-specific information like 'eat at home/on the road' with '-'.
        Additionally, delete any '$' symbols.
        """

        user_message = """\
        Text: {plan}

        Constraint: If transportation details indicate a journey from one city to another (e.g., from A to B), the 'current_city' should be updated to the destination city (in this case, B).
        Use a ';' to separate different attractions, with each attraction formatted as 'Name, City'.
        If there's information about transportation, ensure that the 'current_city' aligns with the destination mentioned in the transportation details (i.e., the current city should follow the format 'from A to B').
        Also, ensure that all flight numbers and costs are followed by a colon (i.e., 'Flight Number:' and 'Cost:'), consistent with the provided example.
        Each item should include ['day', 'current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation'].
        Replace non-specific information like 'eat at home/on the road' with '-'.
        Additionally, delete any '$' symbols.
        -----EXAMPLE-----
        [{{
                "days": 1,
                "current_city": "from Dallas to Peoria",
                "transportation": "Flight Number: 4044830, from Dallas to Peoria, Departure Time: 13:10, Arrival Time: 15:01",
                "breakfast": "-",
                "attraction": "Peoria Historical Society, Peoria;Peoria Holocaust Memorial, Peoria;",
                "lunch": "-",
                "dinner": "Tandoor Ka Zaika, Peoria",
                "accommodation": "Bushwick Music Mansion, Peoria"
            }},
            {{
                "days": 2,
                "current_city": "Peoria",
                "transportation": "-",
                "breakfast": "Tandoor Ka Zaika, Peoria",
                "attraction": "Peoria Riverfront Park, Peoria;The Peoria PlayHouse, Peoria;Glen Oak Park, Peoria;",
                "lunch": "Cafe Hashtag LoL, Peoria",
                "dinner": "The Curzon Room - Maidens Hotel, Peoria",
                "accommodation": "Bushwick Music Mansion, Peoria"
            }},
            {{
                "days": 3,
                "current_city": "from Peoria to Dallas",
                "transportation": "Flight Number: 4045904, from Peoria to Dallas, Departure Time: 07:09, Arrival Time: 09:20",
                "breakfast": "-",
                "attraction": "-",
                "lunch": "-",
                "dinner": "-",
                "accommodation": "-"
            }}]
        -----EXAMPLE END-----
        JSON:
        """

        cutoff = (pred.find("</think>") + 8) if pred.find("</think>") > 0 else 0
        user_message = user_message.format(
            plan=pred[cutoff:]
        )
        prompt = [
            {"role": "user", "content": user_message},
        ]
        logger.debug(system_prompt + "\n" + user_message)
        response = await generate_response(prompt, response_format={"type": "json_object"})
        
        query_data = self.travel_data_dict[query]

        try:
            score, explanation = eval_score(query_data, f'{{"plan":{response}}}')
        except Exception as e:
            score, explanation = 0, str(e)
        
        return score, explanation
    