# Plan-and-Budget

This is the experiment code for the "Plan and Budget: Effective and Efficient Test-Time Scaling on Large Language Model Reasoning" paper.

![overview](imgs/overview.png)

## Environment Setup
To setup the environment, the quickest way is to directly install the `vllm` library, which will automatically install all the required dependency. We also have additional dependencies defined in the `requirements.txt` located under the root directory. Follow the listed commands to install a local conda environment named `vllm`:
```
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm
pip install -r requirements.txt
```
After this, you will need to fill in the local environment variables. They are automatically loaded from a `.env` file. Copy the `.env_template` and rename it to be `.env` file, and fill in the settings. 
> 1. The `API_BASE` usually ends with "/v1". For example, `API_BASE="http://localhost:7878/v1"`. 
> 2. The `API_KEY` is usually `"DUMMY"` unless you are really using the OpenAI API.

> :bulb: **You can choose different environment config file at runtime by prepending an environment variable: `ENV_FILE=path/to/your/.env/file`. This is very useful to manage different LLM models and KG instances.**

## Dataset
Download the [database](https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view?usp=drive_link) and unzip it to the `dataset/TravelPlanner` directory.

The questions in the three datasets are already pre-decomposed, and located underneath the `dataset/MATH-500`, `dataset/NaturalInstruction-Sampled-500` and `dataset/TravelPlanner`.
If you want, you can decompose them again by running the following command, where DATASET_NAME can be `math`, `instruction`, and `travelplanner`. Please adjust the num-workers and queue-size based on your computation resource.
```
python -m dataset.break_down_question --num-workers 32 --queue-size 32 --dataset DATASET_NAME
```

## Reproducing Our Experiment Results
Running the experiment is very straightforward-there is no retraining required, only perform inference on LLM API. Run the following commands in the root directory to reproduce the experiments. Please adjust the num-workers and queue-size based on your computation resource.
```
python -m run.run_inf --num-workers 32 --dataset math --model vanilla
python -m run.run_inf --num-workers 32 --dataset math --model planned
python -m run.run_inf --num-workers 32 --dataset math --model global_budget
python -m run.run_inf --num-workers 32 --dataset math --model planned_global
python -m run.run_inf --num-workers 32 --dataset math --model planned_local_uniform
python -m run.run_inf --num-workers 32 --dataset math --model planned_local_weighted --decay constant --postfix constant
python -m run.run_inf --num-workers 32 --dataset math --model planned_local_weighted --decay linear --postfix linear
python -m run.run_inf --num-workers 32 --dataset math --model planned_local_weighted --decay polynomial --postfix polynomial
python -m run.run_inf --num-workers 32 --dataset math --model planned_local_weighted --decay exponential --postfix exponential
python -m run.run_inf --num-workers 32 --dataset math --model planned_local_weighted --decay cosine --postfix cosine
```
> :bulb: `--postfix` is added to help differentiate them experiment results log files, it doesn't has effect on the experiment results.
> :bulb: If you are curious, you can also tune the allocated budget for different difficulty level in the `utils/utils.py` file.

## Evaluation
For MATH-500 and NaturalInstruction, the evluation is done immediately-you got the final results once the experiment finished. However, TravelPlanner requires to use an external LLM to transform the travel plan into JSON format for evaluation. So, you won't be able to view it after experiment finish (indicated by a -1 score). To evaluate, please run the following command.
```
ENV_FILE=.env.eval python -m run.run_eval --dataset travelplanner --model MODEL_NAME --postfix POSTFIX
```
> :bulb: **Make sure you have choose the correct LLM to evaluate, defined in the `.env.eval` file. The LLM must support JSON structured output.**