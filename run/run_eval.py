import argparse
import asyncio
import json

from inference import *
from utils.logger import *
from utils.utils import *

async def evaluate_all(results):
    loop = asyncio.get_event_loop()
    tasks = [
        loader.post_eval(result['query'], result['prediction'], result['answer'])
        for result in results
    ]
    return await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Evaluation dataset")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_MAP.keys(), help="Model to run inference with")
    parser.add_argument("--postfix", type=str, help="Postfix added to the result file name")
    args = parser.parse_args()

    progress_path = f"results/{args.model}_{args.dataset}_progress{f"_{args.postfix}" if args.postfix else ""}.json"
    output_path = f"results/{args.model}_{args.dataset}_results{f"_{args.postfix}" if args.postfix else ""}.json"

    with open(progress_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    results = results['stats']
    results = sorted(results, key=lambda x: int(x["id"]))

    if args.dataset.lower() == 'travelplanner': # This dataset needs post-evaluation      
        from dataset import *
        loader = TravelPlannerDatasetLoader(config={}, processor=None)
        items = asyncio.run(evaluate_all(results))
        scores = [item[0] for item in items]
        explanations = [item[1] for item in items]
        
        for idx in range(len(results)):
            results[idx]["score"] = scores[idx]
            results[idx]["explanation"] = explanations[idx]

    tokens_sum = 0
    score_sum = 0
    for stat in results:
        score = stat['score']
        tokens = stat['completion_tokens']
        tokens_sum += tokens
        score_sum += score

    score = score_sum / len(results) * 100.0
    avg_tokens = tokens_sum / len(results)
    stats = {
        "len": len(results),
        "score": score,
        "token_sum": tokens_sum,
        "avg_tokens": avg_tokens,
        "score_over_avg_tokens": score / avg_tokens,
        "llm": MODEL_NAME
    }
    results.insert(0, stats)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logger.info(stats)
    logger.info(f"Done evaluation in {args.dataset} dataset on {args.model}_model ✅")