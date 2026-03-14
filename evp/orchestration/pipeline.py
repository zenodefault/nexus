import asyncio
import os
from typing import Dict, List

from evp.agents.hypothesis import HypothesisAgent
from evp.agents.impact_predictor import ImpactPredictorAgent
from evp.agents.literature import LiteratureAgent
from evp.agents.resource_estimator import ResourceEstimatorAgent
from evp.scoring.scoring import score_experiments
from evp.utils.context import RunContext
from evp.utils.llm import LocalLLMClient, MockLLMClient
from evp.utils.logging_utils import get_logger


def get_llm_client():
    mode = os.getenv("EVP_LLM_MODE", "mock").lower()
    if mode == "local":
        return LocalLLMClient()
    return MockLLMClient()


def build_agents(llm_client):
    return {
        "literature": LiteratureAgent(
            name="LiteratureAgent",
            role=LiteratureAgent.role,
            goal=LiteratureAgent.goal,
            prompt_template=LiteratureAgent.prompt_template,
            llm=llm_client,
        ),
        "hypothesis": HypothesisAgent(
            name="HypothesisAgent",
            role=HypothesisAgent.role,
            goal=HypothesisAgent.goal,
            prompt_template=HypothesisAgent.prompt_template,
            llm=llm_client,
        ),
        "resource": ResourceEstimatorAgent(
            name="ResourceEstimatorAgent",
            role=ResourceEstimatorAgent.role,
            goal=ResourceEstimatorAgent.goal,
            prompt_template=ResourceEstimatorAgent.prompt_template,
            llm=llm_client,
        ),
        "impact": ImpactPredictorAgent(
            name="ImpactPredictorAgent",
            role=ImpactPredictorAgent.role,
            goal=ImpactPredictorAgent.goal,
            prompt_template=ImpactPredictorAgent.prompt_template,
            llm=llm_client,
        ),
    }


async def run_pipeline(topic: str, goal: str, constraints: Dict | None = None) -> Dict:
    logger = get_logger("Pipeline")
    context = RunContext(topic=topic, goal=goal, constraints=constraints or {})
    llm = get_llm_client()
    agents = build_agents(llm)

    logger.info("Running LiteratureAgent")
    literature = agents["literature"].run_with_context(context)

    logger.info("Running HypothesisAgent")
    hypothesis = agents["hypothesis"].run_with_context(context)
    experiments: List[Dict] = hypothesis.get("hypotheses", [])

    async def enrich_experiment(exp: Dict) -> Dict:
        resource_task = asyncio.to_thread(
            agents["resource"].run_with_context, context, exp
        )
        impact_task = asyncio.to_thread(
            agents["impact"].run_with_context, context, exp
        )
        resource, impact = await asyncio.gather(resource_task, impact_task)
        return {**exp, **resource, **impact}

    logger.info("Running ResourceEstimatorAgent and ImpactPredictorAgent in parallel")
    enriched = await asyncio.gather(*[enrich_experiment(e) for e in experiments])

    scored = score_experiments(enriched)

    return {
        "topic": topic,
        "goal": goal,
        "literature": literature,
        "hypotheses": experiments,
        "experiments": scored["experiments"],
        "recommended_experiment_id": scored["recommended_experiment_id"],
        "memory": context.memory,
    }
