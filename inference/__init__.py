COMPLETION_LLM = {
    "simplescaling/s1.1-32B",
    "Qwen/QwQ-32B", 
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
}
CHAT_LLM = {
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/llama-3-3-70b-instruct",
    ""
}
REASONING_LLM = {
    "o3-mini-2025-01-31",
    "o4-mini-2025-04-16",
}

from inference._0_vanilla_model import Vanilla_Model
from inference._1_planned_model import Planned_Model
from inference._2_global_budget_model import GlobalBudget_Model
from inference._3_planned_global_model import PlannedGlobal_Model
from inference._4_planned_local_uniform_model import PlannedLocalUniform_Model
from inference._5_planned_local_weighted_model import PlannedLocalWeighted_Model

MODEL_MAP = {
    "vanilla": Vanilla_Model,
    "planned": Planned_Model,
    "global_budget": GlobalBudget_Model,
    "planned_global": PlannedGlobal_Model,
    "planned_local_uniform": PlannedLocalUniform_Model,
    "planned_local_weighted": PlannedLocalWeighted_Model,
}