import os
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
import logging
from enum import Enum
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from functools import partial
from pathlib import Path
import json

import openai
from langchain.tools import DuckDuckGoSearchRun
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentRole(Enum):
    ADVOCATE = "advocate"
    CRITIC = "critic"
    ARBITRATOR = "arbitrator"

@dataclass
class Fact:
    content: str
    confidence: float = 0.0
    evidence: List[str] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

class DatasetLoader:
    """Loader for multiple misinformation datasets"""
    
    DATASETS_CONFIG = {
        'antivax': {
            'path': 'ComplexDataLab/Misinfo_Datasets',
            'splits': {
                'train': 'antivax/antivax_train.parquet',
                'test': 'antivax/antivax_test.parquet',
                'validation': 'antivax/antivax_validation.parquet'
            }
        },
        'fever': {
            'path': 'fever-2021-fact',
            'splits': {
                'train': 'train.jsonl',
                'test': 'test.jsonl'
            }
        },
        # Add other datasets configs here
    }
    
    @staticmethod
    def load_dataset(name: str, split: str) -> pd.DataFrame:
        """Load a specific dataset split"""
        config = DatasetLoader.DATASETS_CONFIG.get(name)
        if not config:
            raise ValueError(f"Dataset {name} not found in config")
            
        if name == 'antivax':
            return pd.read_parquet(f"hf://datasets/{config['path']}/{config['splits'][split]}")
        elif name == 'fever':
            return pd.read_json(f"{config['path']}/{config['splits'][split]}", lines=True)
        # Add other dataset loading logic here
        
        raise NotImplementedError(f"Loading logic for {name} not implemented")

class DoubleDecompositionAgent:
    def __init__(self, openai_api_key: str, batch_size: int = 32, max_workers: int = 4):
        self.search = DuckDuckGoSearchRun()
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.2,
            openai_api_key=openai_api_key
        )
        
        # Initialize prompt templates
        self.prompts = self._load_prompts()
        self._init_prompts()
        self.batch_size = batch_size
        self.max_workers = max_workers
        
    def _load_prompts(self) -> Dict:
        """Load prompts from YAML file"""
        import yaml
        with open('prompt.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def _init_prompts(self):
        """Initialize prompt templates for different stages"""
        self.five_w1h_prompt = PromptTemplate(
            input_variables=["claim"],
            template=self.prompts['five_w1h']['user']
        )
        
        self.atomic_facts_prompt = PromptTemplate(
            input_variables=["component"],
            template=self.prompts['atomic_facts']['user']
        )

    async def decompose_claim(self, claim: str) -> Dict:
        """Perform primary 5W1H decomposition"""
        chain = LLMChain(llm=self.llm, prompt=self.five_w1h_prompt)
        components = await chain.arun(claim=claim)
        return self._parse_5w1h(components)
    
    async def extract_atomic_facts(self, component: str) -> List[Fact]:
        """Perform secondary decomposition into atomic facts"""
        chain = LLMChain(llm=self.llm, prompt=self.atomic_facts_prompt)
        facts_text = await chain.arun(component=component)
        return self._parse_atomic_facts(facts_text)

    def _query_priority(self, query: str, info_gain: float, 
                       importance: float) -> float:
        """Calculate search priority score for a query"""
        # Implement priority scoring based on expected information gain,
        # verification importance, and cost considerations
        search_feasibility = 0.8  # Simplified feasibility score
        cost = len(query.split()) * 0.1  # Simple cost heuristic
        
        return (0.4 * info_gain + 0.3 * importance + 
                0.2 * search_feasibility - 0.1 * cost)

    async def verify_fact(self, fact: Fact) -> Tuple[bool, List[str]]:
        """Verify a single atomic fact using search"""
        try:
            search_results = self.search.run(fact.content)
            
            # Use LLM to analyze search results
            verification_prompt = f"""
            Fact to verify: {fact.content}
            Evidence: {search_results}
            
            Determine if the evidence supports or contradicts the fact.
            Return: true/false and list key supporting/contradicting points.
            """
            
            response = await self.llm.agenerate([[HumanMessage(content=verification_prompt)]])
            # Parse response and update fact confidence
            return self._parse_verification_result(response.generations[0][0].text)
            
        except Exception as e:
            logger.error(f"Error verifying fact: {e}")
            return False, []

    async def run_verification(self, claim: str) -> Dict:
        """Main verification pipeline"""
        results = {
            "claim": claim,
            "components": {},
            "final_score": 0.0,
            "explanation": ""
        }
        
        # Step 1: Primary decomposition (5W1H)
        components = await self.decompose_claim(claim)
        
        # Step 2: Secondary decomposition (atomic facts)
        all_facts = []
        for category, component in components.items():
            facts = await self.extract_atomic_facts(component)
            results["components"][category] = {
                "content": component,
                "facts": facts
            }
            all_facts.extend(facts)
            
        # Step 3: Verify facts with multi-agent approach
        await self._run_multi_agent_verification(all_facts, results)
        
        return results

    async def _run_multi_agent_verification(self, facts: List[Fact], 
                                          results: Dict) -> None:
        """Run multi-agent verification process"""
        advocate_evidence = []
        critic_evidence = []
        
        # Advocate agent gathers supporting evidence
        for fact in facts:
            supported, evidence = await self.verify_fact(fact)
            if supported:
                advocate_evidence.extend(evidence)
                
        # Critic agent looks for contradicting evidence
        critic_prompt = self._generate_critic_prompt(facts, advocate_evidence)
        critic_response = await self.llm.agenerate([[HumanMessage(content=critic_prompt)]])
        critic_evidence = self._parse_critic_response(critic_response.generations[0][0].text)
        
        # Arbitrator makes final decision
        final_score, explanation = await self._arbitrate(
            facts, advocate_evidence, critic_evidence
        )
        
        results["final_score"] = final_score
        results["explanation"] = explanation
        results["advocate_evidence"] = advocate_evidence
        results["critic_evidence"] = critic_evidence

    def _generate_critic_prompt(self, facts: List[Fact], 
                              advocate_evidence: List[str]) -> str:
        """Generate prompt for critic agent"""
        return f"""
        Analyze these claims and evidence for potential flaws or contradictions:
        
        Claims:
        {[f.content for f in facts]}
        
        Supporting Evidence:
        {advocate_evidence}
        
        Identify any logical fallacies, missing context, or contradicting evidence.
        """

    async def _arbitrate(self, facts: List[Fact], advocate_evidence: List[str],
                        critic_evidence: List[str]) -> Tuple[float, str]:
        """Arbitrator agent makes final decision"""
        arbitration_prompt = f"""
        Review all evidence and make a final decision:
        
        Claims:
        {[f.content for f in facts]}
        
        Supporting Evidence:
        {advocate_evidence}
        
        Contradicting Evidence:
        {critic_evidence}
        
        Provide:
        1. Confidence score (0-1)
        2. Detailed explanation of decision
        """
        
        response = await self.llm.agenerate([[HumanMessage(content=arbitration_prompt)]])
        return self._parse_arbitration_result(response.generations[0][0].text)

    def _parse_5w1h(self, components: str) -> Dict:
        """Parse 5W1H analysis results"""
        import json
        try:
            result = json.loads(components)
            return {
                'who': result.get('who', ''),
                'what': result.get('what', ''),
                'when': result.get('when', ''),
                'where': result.get('where', ''),
                'why': result.get('why', ''),
                'how': result.get('how', '')
            }
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing 5W1H results: {e}")
            return {}

    def _parse_atomic_facts(self, facts_text: str) -> List[Fact]:
        """Parse atomic facts from text"""
        import json
        try:
            facts_data = json.loads(facts_text)
            return [
                Fact(
                    content=fact['content'],
                    confidence=fact.get('confidence', 0.0)
                ) for fact in facts_data
            ]
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing atomic facts: {e}")
            return []

    def _parse_verification_result(self, result: str) -> Tuple[bool, List[str]]:
        """Parse verification results"""
        import json
        try:
            data = json.loads(result)
            return (
                data.get('supported', False),
                data.get('evidence_points', [])
            )
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing verification result: {e}")
            return False, []

    def _parse_critic_response(self, response: str) -> List[str]:
        """Parse critic agent response"""
        import json
        try:
            data = json.loads(response)
            evidence = []
            evidence.extend(data.get('logical_fallacies', []))
            evidence.extend(data.get('missing_context', []))
            evidence.extend(data.get('contradictions', []))
            return evidence
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing critic response: {e}")
            return []

    def _parse_arbitration_result(self, result: str) -> Tuple[float, str]:
        """Parse arbitration results"""
        import json
        try:
            data = json.loads(result)
            return (
                float(data.get('score', 0.0)),
                data.get('explanation', '')
            )
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing arbitration result: {e}")
            return 0.0, ""

    async def evaluate(self, 
                      data: Union[str, pd.DataFrame],
                      dataset_name: Optional[str] = None,
                      split: Optional[str] = None,
                      cache_dir: Optional[str] = None) -> Dict:
        """
        Evaluate on dataset or single claim
        
        Args:
            data: Either a string (single claim) or DataFrame (batch evaluation)
            dataset_name: Name of dataset if loading from config
            split: Dataset split if loading from config
            cache_dir: Directory to cache results
        """
        if isinstance(data, str):
            return await self.run_verification(data)
            
        if dataset_name and split:
            data = DatasetLoader.load_dataset(dataset_name, split)
            
        return await self._batch_evaluate(data, cache_dir)
    
    async def _batch_evaluate(self, df: pd.DataFrame, cache_dir: Optional[str] = None) -> Dict:
        """Run batch evaluation on DataFrame"""
        results = []
        cache = {}
        
        if cache_dir:
            cache_path = Path(cache_dir) / "verification_cache.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    cache = json.load(f)
        
        # Split into batches
        batches = [df[i:i + self.batch_size] for i in range(0, len(df), self.batch_size)]
        
        async def process_batch(batch):
            batch_results = []
            for _, row in batch.iterrows():
                claim = row['claim']
                if claim in cache:
                    result = cache[claim]
                else:
                    try:
                        result = await self.run_verification(claim)
                        if cache_dir:
                            cache[claim] = result
                    except Exception as e:
                        logger.error(f"Error processing claim: {claim}")
                        logger.error(e)
                        result = {
                            "claim": claim,
                            "error": str(e),
                            "final_score": 0.0
                        }
                batch_results.append(result)
            return batch_results
                
        # Process batches concurrently
        for batch in tqdm(batches, desc="Processing batches"):
            batch_results = await process_batch(batch)
            results.extend(batch_results)
            
            # Update cache file
            if cache_dir:
                with open(cache_path, 'w') as f:
                    json.dump(cache, f)
                    
        # Calculate metrics
        metrics = self._calculate_metrics(df, results)
        
        return {
            "results": results,
            "metrics": metrics
        }
    
    def _calculate_metrics(self, df: pd.DataFrame, results: List[Dict]) -> Dict:
        """Calculate evaluation metrics"""
        predictions = [r['final_score'] > 0.5 for r in results]
        labels = df['label'].values
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions),
            "recall": recall_score(labels, predictions),
            "f1": f1_score(labels, predictions)
        }

if __name__ == "__main__":
    # Example usage
    api_key = os.getenv("OPENAI_API_KEY")
    agent = DoubleDecompositionAgent(api_key)
    
    claim = "79.4% of babies who die of 'SIDS' had a vaccine the same day."
    
    import asyncio
    
    # Single claim
    results = asyncio.run(agent.evaluate(claim))
    print("Single claim results:", results)
    
    # Batch evaluation
    results = asyncio.run(agent.evaluate(
        dataset_name="antivax",
        split="validation",
        cache_dir="./cache"
    ))
    print("\nBatch evaluation metrics:", results["metrics"])
