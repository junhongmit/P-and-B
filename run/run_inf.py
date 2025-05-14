import argparse
import functools

from inference import *
from utils.logger import *
from utils.utils import *

if __name__ == "__main__":
    from dataset import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Evaluation dataset")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_MAP.keys(), help="Model to run inference with")
    parser.add_argument("--num-workers", type=int, default=500, help="Number of workers generating the answers")
    parser.add_argument("--queue-size", type=int, default=500, help="Queue size of data loading")
    parser.add_argument("--decay", type=str, choices=["constant", "linear", "polynomial", "exponential", "cosine"], help="Weighted model config")
    parser.add_argument("--postfix", type=str, help="Postfix added to the result file name")
    parser.add_argument("--keep", action="store_true", help="Keep the progress file")
    args = parser.parse_args()

    config = {
        "num_workers": args.num_workers,
        "queue_size": args.queue_size,
        "decomposed": True
    }

    participant_model = MODEL_MAP[args.model](decay=args.decay)

    progress_path = f"results/{participant_model.name}_{args.dataset}_progress{f"_{args.postfix}" if args.postfix else ""}.json"
    result_path = f"results/{participant_model.name}_{args.dataset}_results{f"_{args.postfix}" if args.postfix else ""}.json"
    logger = QAProgressLogger(progress_path=progress_path)
    participant_model.logger = logger
    print(logger.processed_questions)

    loader = None
    if args.dataset.lower() == 'math':
        loader = MathDatasetLoader(
            config=config, 
            logger=logger,
            processor=functools.partial(participant_model.process_question, logger=logger)
        )
    elif args.dataset.lower() == 'instruction':
        loader = InstructionDatasetLoader(
            config=config, 
            logger=logger,
            processor=functools.partial(participant_model.process_question, logger=logger)
        )
    elif args.dataset.lower() == 'travelplanner':
        loader = TravelPlannerDatasetLoader(
            config=config, 
            logger=logger,
            processor=functools.partial(participant_model.process_question, logger=logger)
        )
    else:
        raise NotImplementedError("Dataset is not supported ❌")

    participant_model.loader = loader
    participant_model.instruction = loader._instruction_prompt
    participant_model.output_format = loader._output_format
    participant_model.evaluator = loader.eval
    
    loop = always_get_an_event_loop()
    loop.run_until_complete(
        loader.run()
    )

    results = [
        {"id": int(stat["id"]), "query": stat["query"], "answer": stat["answer"], 
        "prediction": stat["prediction"], "prompt_tokens": stat["prompt_tokens"],
        "completion_tokens": stat["completion_tokens"], "total_tokens": stat["total_tokens"],
        "score": stat["score"], "explanation": stat["explanation"]}
        for stat in logger.progress_data["stats"]
    ]
    results = sorted(results, key=lambda x: x["id"])
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

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

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    if not args.keep:
        os.remove(progress_path)

    logger.info(stats)
    logger.info(f"Done inference in {args.dataset} dataset on {args.model}_model ✅")