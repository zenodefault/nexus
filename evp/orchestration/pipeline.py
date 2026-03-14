import asyncio
import os
from typing import Dict, List

from evp.agents.hypothesis import HypothesisAgent
from evp.agents.impact_predictor import ImpactPredictorAgent
from evp.agents.literature import LiteratureAgent
from evp.agents.resource_estimator import ResourceEstimatorAgent
from evp.data.arxiv import build_literature_digest, fetch_papers
from evp.scoring.scoring import score_experiments
from evp.utils.context import RunContext
from evp.utils.llm import LocalLLMClient, MockLLMClient
from evp.utils.logging_utils import get_logger


def _current_mode() -> str:
    return os.getenv("EVP_LLM_MODE", "mock").lower()


def get_llm_client():
    mode = _current_mode()
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
    mode = _current_mode()
    if mode == "mock_static":
        return _static_mock_pipeline(topic, goal)

    context = RunContext(topic=topic, goal=goal, constraints=constraints or {})

    papers = _load_papers_for_context(topic, mode, logger)
    context.constraints["papers"] = papers
    context.constraints["literature_digest"] = build_literature_digest(papers)

    llm = get_llm_client()
    agents = build_agents(llm)

    logger.info("Running LiteratureAgent")
    literature = agents["literature"].run_with_context(context)

    logger.info("Running HypothesisAgent")
    hypothesis = agents["hypothesis"].run_with_context(context)
    experiments: List[Dict] = hypothesis.get("hypotheses", [])

    async def enrich_experiment(exp: Dict) -> Dict:
        async def run_resource() -> Dict:
            await asyncio.sleep(0)
            return agents["resource"].run_with_context(context, exp)

        async def run_impact() -> Dict:
            await asyncio.sleep(0)
            return agents["impact"].run_with_context(context, exp)

        resource_task = asyncio.create_task(run_resource())
        impact_task = asyncio.create_task(run_impact())
        resource, impact = await asyncio.gather(resource_task, impact_task)
        return {**exp, **resource, **impact}

    logger.info("Running ResourceEstimatorAgent and ImpactPredictorAgent in parallel")
    enriched = await asyncio.gather(*[enrich_experiment(e) for e in experiments])

    scored = score_experiments(enriched)

    return {
        "topic": topic,
        "goal": goal,
        "literature": literature,
        "papers": papers,
        "hypotheses": experiments,
        "experiments": scored["experiments"],
        "recommended_experiment_id": scored["recommended_experiment_id"],
        "memory": context.memory,
    }


def _load_papers_for_context(topic: str, mode: str, logger) -> List[Dict]:
    if mode == "mock":
        return _mock_papers(topic)
    if os.getenv("EVP_USE_ARXIV", "true").lower() != "true":
        return []
    try:
        return fetch_papers(topic)
    except Exception as exc:
        logger.warning("Unable to fetch arXiv papers: %s", exc)
        return []


def _mock_papers(topic: str) -> List[Dict]:
    return [
        {
            "paper_id": "2501.00001",
            "title": f"Contrastive Learning for {topic} Classification",
            "abstract": "Contrastive pretraining improves representation quality on limited labeled data.",
            "authors": ["A. Kumar", "M. Chen"],
            "published": "2025-01-03T00:00:00",
            "updated": "2025-01-10T00:00:00",
            "url": "https://arxiv.org/abs/2501.00001",
            "categories": ["cs.CV"],
        },
        {
            "paper_id": "2502.00002",
            "title": f"EfficientNet Baselines for {topic} with Limited Labels",
            "abstract": "EfficientNet offers strong performance with low compute budgets.",
            "authors": ["R. Singh"],
            "published": "2025-02-05T00:00:00",
            "updated": "2025-02-05T00:00:00",
            "url": "https://arxiv.org/abs/2502.00002",
            "categories": ["cs.LG"],
        },
    ]


def _static_mock_pipeline(topic: str, goal: str) -> Dict:
    experiments = [
        {
            "id": "exp_1",
            "title": "ViT with transfer learning",
            "model": "Vision Transformer",
            "compute_units": "High",
            "resource_rationale": "ViT fine-tuning typically needs larger memory footprints.",
            "novelty_score": 8,
            "expected_gain": 3.0,
            "impact_rationale": "Strong performance upside if data volume is sufficient.",
        },
        {
            "id": "exp_2",
            "title": "CNN with contrastive learning",
            "model": "CNN + contrastive",
            "compute_units": "Medium",
            "resource_rationale": "Moderate contrastive pretraining plus supervised fine-tuning.",
            "novelty_score": 7,
            "expected_gain": 2.5,
            "impact_rationale": "Balanced improvement potential with practical compute.",
        },
        {
            "id": "exp_3",
            "title": "EfficientNet baseline",
            "model": "EfficientNet",
            "compute_units": "Low",
            "resource_rationale": "Efficient architecture with lower training cost.",
            "novelty_score": 5,
            "expected_gain": 1.5,
            "impact_rationale": "Lower novelty but fast and inexpensive to validate.",
        },
    ]
    scored = score_experiments(experiments)
    return {
        "topic": topic,
        "goal": goal,
        "literature": {
            "summary": "Static mock mode enabled for frontend testing.",
            "key_findings": [],
            "limitations": [],
            "hypotheses": [],
        },
        "papers": _mock_papers(topic),
        "hypotheses": [
            {"id": e["id"], "title": e["title"], "model": e["model"]} for e in experiments
        ],
        "experiments": scored["experiments"],
        "recommended_experiment_id": scored["recommended_experiment_id"],
        "memory": [],
    }
