"""
Benchmarks module for evaluating LLMs on insurance domain tasks.

This module provides benchmark datasets and utilities for testing
LLM performance on insurance-specific tasks.
"""

import logging
import json
import os
import csv
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import numpy as np

from .metrics import EvaluationMetrics, get_metrics_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_BENCHMARKS_DIR = os.path.join(os.path.dirname(__file__), "benchmarks")

@dataclass
class BenchmarkExample:
    """Class representing a single benchmark example."""

    id: str
    input_text: str
    reference_output: str
    metadata: Dict[str, Any]

@dataclass
class Benchmark:
    """Class representing a benchmark dataset for an insurance task."""

    name: str
    task_type: str
    description: str
    examples: List[BenchmarkExample]
    metrics: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark to dictionary for serialization."""
        return {
            "name": self.name,
            "task_type": self.task_type,
            "description": self.description,
            "examples": [asdict(ex) for ex in self.examples],
            "metrics": self.metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Benchmark':
        """Create a benchmark from a dictionary."""
        examples = [
            BenchmarkExample(
                id=ex["id"],
                input_text=ex["input_text"],
                reference_output=ex["reference_output"],
                metadata=ex["metadata"]
            )
            for ex in data["examples"]
        ]

        return cls(
            name=data["name"],
            task_type=data["task_type"],
            description=data["description"],
            examples=examples,
            metrics=data["metrics"]
        )

    def save(self, output_path: str):
        """
        Save the benchmark to a JSON file.

        Args:
            output_path: Path to save the benchmark
        """
        try:
            with open(output_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Saved benchmark to {output_path}")
        except Exception as e:
            logger.error(f"Error saving benchmark to {output_path}: {str(e)}")

    @classmethod
    def load(cls, input_path: str) -> 'Benchmark':
        """
        Load a benchmark from a JSON file.

        Args:
            input_path: Path to load the benchmark from

        Returns:
            Loaded benchmark
        """
        try:
            with open(input_path, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded benchmark from {input_path}")
            return cls.from_dict(data)
        except Exception as e:
            logger.error(
                f"Error loading benchmark from {input_path}: {str(e)}")
            raise

@dataclass
class BenchmarkResult:
    """Class representing the result of running a benchmark."""

    benchmark_name: str
    model_id: str
    task_type: str
    results: List[Dict[str, Any]]
    aggregate_scores: Dict[str, float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create a benchmark result from a dictionary."""
        return cls(**data)

    def save(self, output_path: str):
        """
        Save the benchmark result to a JSON file.

        Args:
            output_path: Path to save the result
        """
        try:
            with open(output_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Saved benchmark result to {output_path}")
        except Exception as e:
            logger.error(
                f"Error saving benchmark result to {output_path}: {str(e)}")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert benchmark results to a pandas DataFrame.

        Returns:
            DataFrame with benchmark results
        """
        return pd.DataFrame(self.results)

class BenchmarkManager:
    """Manager class for benchmarks and benchmark runs."""

    def __init__(self, benchmarks_dir: Optional[str] = None):
        """
        Initialize the benchmark manager.

        Args:
            benchmarks_dir: Directory for benchmark datasets
        """
        self.benchmarks_dir = benchmarks_dir or DEFAULT_BENCHMARKS_DIR
        Path(self.benchmarks_dir).mkdir(parents=True, exist_ok=True)
        self.metrics_manager = get_metrics_manager()
        self.benchmarks: Dict[str, Benchmark] = {}
        self._load_benchmarks()

    def _load_benchmarks(self):
        """Load existing benchmarks from disk."""
        benchmarks_dir = Path(self.benchmarks_dir)

        for file_path in benchmarks_dir.glob("*.json"):
            try:
                benchmark = Benchmark.load(str(file_path))
                self.benchmarks[benchmark.name] = benchmark
                logger.info(f"Loaded benchmark: {benchmark.name}")
            except Exception as e:
                logger.error(
                    f"Error loading benchmark from {file_path}: {str(e)}")

        logger.info(f"Loaded {len(self.benchmarks)} benchmarks")

    def create_benchmark(
        self,
        name: str,
        task_type: str,
        description: str,
        examples: List[BenchmarkExample],
        metrics: Optional[List[str]] = None
    ) -> Benchmark:
        """
        Create a new benchmark.

        Args:
            name: Name of the benchmark
            task_type: Type of insurance task
            description: Description of the benchmark
            examples: List of benchmark examples
            metrics: List of metrics to use for evaluation

        Returns:
            The created benchmark
        """

        if metrics is None:
            if task_type == "policy_summary":
                metrics = ["rouge", "bleu", "relevance", "completeness"]
            elif task_type == "claim_response":
                metrics = ["relevance", "compliance"]
            elif task_type == "customer_communication":
                metrics = ["relevance", "complexity"]
            elif task_type == "risk_assessment":
                metrics = ["relevance", "completeness", "compliance"]
            else:
                metrics = ["rouge", "relevance"]

        benchmark = Benchmark(
            name=name,
            task_type=task_type,
            description=description,
            examples=examples,
            metrics=metrics
        )

        self.benchmarks[name] = benchmark

        benchmark_path = Path(self.benchmarks_dir) / f"{name}.json"
        benchmark.save(str(benchmark_path))

        return benchmark

    def get_benchmark(self, name: str) -> Optional[Benchmark]:
        """
        Get a benchmark by name.

        Args:
            name: Name of the benchmark

        Returns:
            The benchmark or None if not found
        """
        return self.benchmarks.get(name)

    def list_benchmarks(self) -> List[Dict[str, str]]:
        """
        List all available benchmarks.

        Returns:
            List of dictionaries with benchmark information
        """
        return [
            {
                "name": b.name,
                "task_type": b.task_type,
                "description": b.description,
                "num_examples": len(b.examples)
            }
            for b in self.benchmarks.values()
        ]

    def run_benchmark(
        self,
        benchmark_name: str,
        model_id: str,
        generate_fn: Callable[[str], str],
        metrics: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Run a benchmark on a model.

        Args:
            benchmark_name: Name of the benchmark to run
            model_id: ID of the model being evaluated
            generate_fn: Function that takes input text and returns generated text
            metrics: Optional list of metrics to use (overrides benchmark metrics)
            context: Additional context for evaluation

        Returns:
            Benchmark result with evaluation scores
        """
        benchmark = self.get_benchmark(benchmark_name)
        if not benchmark:
            raise ValueError(f"Benchmark not found: {benchmark_name}")

        metrics = metrics or benchmark.metrics

        available_metrics = [m["name"]
                             for m in self.metrics_manager.list_metrics()]
        for metric in metrics:
            if metric not in available_metrics:
                logger.warning(f"Metric not available: {metric}")

        results = []

        for example in benchmark.examples:
            try:

                generated_text = generate_fn(example.input_text)

                evaluation_context = {
                    **(context or {}),
                    **example.metadata
                }

                metric_results = self.metrics_manager.evaluate(
                    generated_text=generated_text,
                    metric_names=metrics,
                    reference_text=example.reference_output,
                    context=evaluation_context
                )

                result = {
                    "example_id": example.id,
                    "input_text": example.input_text,
                    "reference_output": example.reference_output,
                    "generated_text": generated_text,
                    "metrics": {
                        name: result.score
                        for name, result in metric_results.items()
                    },
                    "details": {
                        name: result.details
                        for name, result in metric_results.items()
                    }
                }

                results.append(result)

            except Exception as e:
                logger.error(
                    f"Error evaluating example {example.id}: {str(e)}")

                results.append({
                    "example_id": example.id,
                    "input_text": example.input_text,
                    "reference_output": example.reference_output,
                    "generated_text": "",
                    "metrics": {},
                    "details": {"error": str(e)}
                })

        aggregate_scores = {}

        for metric in metrics:
            scores = [r["metrics"].get(metric, 0)
                      for r in results if metric in r["metrics"]]
            if scores:
                aggregate_scores[metric] = sum(scores) / len(scores)
            else:
                aggregate_scores[metric] = 0.0

        if aggregate_scores:
            aggregate_scores["overall"] = sum(
                aggregate_scores.values()) / len(aggregate_scores)
        else:
            aggregate_scores["overall"] = 0.0

        benchmark_result = BenchmarkResult(
            benchmark_name=benchmark_name,
            model_id=model_id,
            task_type=benchmark.task_type,
            results=results,
            aggregate_scores=aggregate_scores,
            metadata={
                "num_examples": len(benchmark.examples),
                "successful_evaluations": sum(1 for r in results if r["metrics"]),
                "timestamp": pd.Timestamp.now().isoformat()
            }
        )

        results_dir = Path(self.benchmarks_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        result_path = results_dir / \
            f"{benchmark_name}_{model_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        benchmark_result.save(str(result_path))

        return benchmark_result

    def compare_models(
        self,
        benchmark_name: str,
        model_results: Dict[str, BenchmarkResult]
    ) -> pd.DataFrame:
        """
        Compare multiple models on a benchmark.

        Args:
            benchmark_name: Name of the benchmark
            model_results: Dictionary mapping model IDs to benchmark results

        Returns:
            DataFrame with comparison results
        """
        if not model_results:
            return pd.DataFrame()

        task_type = next(iter(model_results.values())).task_type

        comparison_data = []

        for model_id, result in model_results.items():
            row = {
                "model_id": model_id,
                "task_type": task_type,
                "benchmark": benchmark_name,
                "overall_score": result.aggregate_scores["overall"]
            }

            for metric, score in result.aggregate_scores.items():
                if metric != "overall":
                    row[f"{metric}_score"] = score

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def export_results_to_csv(
        self,
        benchmark_results: List[BenchmarkResult],
        output_path: str
    ) -> bool:
        """
        Export benchmark results to CSV.

        Args:
            benchmark_results: List of benchmark results
            output_path: Path to save the CSV file

        Returns:
            True if successful, False otherwise
        """
        if not benchmark_results:
            logger.warning("No benchmark results to export")
            return False

        try:

            all_data = []

            for result in benchmark_results:
                for example_result in result.results:
                    row = {
                        "benchmark": result.benchmark_name,
                        "model_id": result.model_id,
                        "task_type": result.task_type,
                        "example_id": example_result["example_id"]
                    }

                    for metric, score in example_result["metrics"].items():
                        row[f"{metric}_score"] = score

                    all_data.append(row)

            df = pd.DataFrame(all_data)
            df.to_csv(output_path, index=False)

            logger.info(f"Exported benchmark results to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting to CSV {output_path}: {str(e)}")
            return False

def get_benchmark_manager() -> BenchmarkManager:
    """Get or create a global benchmark manager instance."""
    return BenchmarkManager()

def create_policy_summary_benchmark() -> Benchmark:
    """
    Create a sample benchmark for policy summarization.

    Returns:
        A benchmark for policy summarization
    """
    examples = [
        BenchmarkExample(
            id="policy_summary_001",
            input_text="""
AUTO INSURANCE POLICY
Policy Number: AP-12345678

Named Insured: John Smith
Address: 123 Main Street, Anytown, USA 12345
Policy Period: 01/01/2023 to 01/01/2024 (12:01 AM standard time)

INSURED VEHICLE:
2020 Toyota Camry, VIN: 1HGCM82633A123456

COVERAGES AND LIMITS:
Part A - LIABILITY
Bodily Injury: $100,000 per person / $300,000 per accident
Property Damage: $50,000 per accident
Premium: $650.00

Part B - MEDICAL PAYMENTS
$5,000 per person
Premium: $120.00

Part C - UNINSURED MOTORIST
Bodily Injury: $100,000 per person / $300,000 per accident
Premium: $85.00

Part D - PHYSICAL DAMAGE
Comprehensive: $500 deductible
Premium: $220.00
Collision: $500 deductible
Premium: $380.00

Total Premium: $1,455.00

EXCLUSIONS:
This policy does not provide coverage for:
a. Intentional damage caused by you or a family member
b. Damage to property owned by you or a family member
c. Using the vehicle for delivery services or ride-sharing without proper endorsement
d. Racing or speed contests
e. Vehicle used for business purposes unless declared
            """,
            reference_output="""
This auto insurance policy for John Smith (Policy Number AP-12345678) covers a 2020 Toyota Camry from 01/01/2023 to 01/01/2024. It provides liability coverage ($100,000 per person/$300,000 per accident for bodily injury; $50,000 per accident for property damage), medical payments ($5,000 per person), uninsured motorist coverage ($100,000 per person/$300,000 per accident), and physical damage protection (comprehensive and collision with $500 deductibles each). The total premium is $1,455.00. Key exclusions include intentional damage, damage to owned property, delivery/ride-sharing use without endorsement, racing, and business use unless declared.
            """.strip(),
            metadata={
                "required_sections": ["coverages", "limits", "exclusions", "premium"],
                "policy_type": "auto",
                "required_phrases": ["deductible", "coverage", "exclusions"],
                "prohibited_phrases": ["not sure", "can't determine"]
            }
        ),
        BenchmarkExample(
            id="policy_summary_002",
            input_text="""
HOMEOWNER'S INSURANCE POLICY
Policy Number: HP-87654321

Named Insured: Jane Doe
Address: 456 Elm Street, Anytown, USA 12345
Property Address: Same as above
Policy Period: 03/15/2023 to 03/15/2024 (12:01 AM standard time)

COVERAGES AND LIMITS:

Section I - Property Coverages
A. Dwelling: $350,000
B. Other Structures: $35,000 (10% of Dwelling)
C. Personal Property: $175,000 (50% of Dwelling)
D. Loss of Use: $70,000 (20% of Dwelling)

Section II - Liability Coverages
E. Personal Liability: $300,000 per occurrence
F. Medical Payments to Others: $5,000 per person

Deductibles:
All perils: $1,000
Wind/Hail: $2,500

Premium: $1,250.00 annually

EXCLUSIONS:
This policy does not provide coverage for:
1. Earth movement (earthquake, landslide)
2. Water damage from flood or surface water
3. Neglect or intentional loss
4. War or nuclear hazard
5. Business activities conducted on premises
6. Mold damage (limited coverage available)
            """,
            reference_output="""
This homeowner's insurance policy for Jane Doe (Policy Number HP-87654321) covers the property at 456 Elm Street from 03/15/2023 to 03/15/2024. Property coverages include $350,000 for the dwelling, $35,000 for other structures, $175,000 for personal property, and $70,000 for loss of use. Liability coverages include $300,000 per occurrence for personal liability and $5,000 per person for medical payments to others. Deductibles are $1,000 for all perils and $2,500 for wind/hail damage. The annual premium is $1,250.00. Key exclusions include earth movement, flood damage, neglect or intentional loss, war or nuclear hazard, business activities on premises, and mold damage (with limited coverage available).
            """.strip(),
            metadata={
                "required_sections": ["coverages", "limits", "exclusions", "premium", "deductibles"],
                "policy_type": "homeowner",
                "required_phrases": ["deductible", "coverage", "exclusions", "liability"],
                "prohibited_phrases": ["not sure", "can't determine"]
            }
        )
    ]

    benchmark = Benchmark(
        name="policy_summary_benchmark",
        task_type="policy_summary",
        description="Benchmark for evaluating insurance policy summarization",
        examples=examples,
        metrics=["rouge", "bleu", "relevance", "completeness", "compliance"]
    )

    return benchmark

def create_claim_response_benchmark() -> Benchmark:
    """
    Create a sample benchmark for claim response generation.

    Returns:
        A benchmark for claim response generation
    """
    examples = [
        BenchmarkExample(
            id="claim_response_001",
            input_text="""
CLAIM INFORMATION:
Claim Number: CL-87654321
Policyholder: John Smith
Policy Number: AP-12345678
Date of Incident: 05/15/2023
Type of Claim: Auto accident
Description: Policyholder was stopped at a red light when another vehicle rear-ended them. Damage to rear bumper and trunk. No injuries reported.
Damage Estimate: $2,800
Coverage: Collision with $500 deductible

TASK:
Draft a claim approval letter to the policyholder.
            """,
            reference_output="""
Dear Mr. Smith,

Re: Claim Number CL-87654321, Policy Number AP-12345678

Thank you for submitting your claim regarding the auto accident that occurred on May 15, 2023. We have completed our review of your claim for damage to your vehicle's rear bumper and trunk.

I am pleased to inform you that your claim has been approved under the collision coverage of your policy. Based on the damage estimate of $2,800 and your policy's $500 deductible, we have approved a payment in the amount of $2,300.

A check will be mailed to your address on file within 5-7 business days. Alternatively, if you've signed up for direct deposit, the funds will be transferred to your account within 3 business days.

If you have chosen to use one of our preferred repair shops, you may authorize them to begin work immediately, and we will coordinate payment directly with them. If you select a repair shop outside our network, please provide them with your claim information, and they can contact us with any questions.

Should you have any questions about your claim or the payment, please don't hesitate to contact our claims department at (555) 123-4567 or claims@abcinsurance.com.

Thank you for your business. We value you as a customer and are committed to providing prompt and fair claim settlements.

Sincerely,

Claims Representative
ABC Insurance Company
            """.strip(),
            metadata={
                "claim_type": "auto",
                "response_type": "approval",
                "required_phrases": ["approved", "deductible", "payment"],
                "prohibited_phrases": ["denied", "rejection", "insufficient", "not covered"]
            }
        ),
        BenchmarkExample(
            id="claim_response_002",
            input_text="""
CLAIM INFORMATION:
Claim Number: CL-12398765
Policyholder: Jane Doe
Policy Number: HP-87654321
Date of Incident: 06/10/2023
Type of Claim: Water damage from roof leak
Description: Policyholder reports water damage to ceiling and walls in upstairs bathroom due to roof leak during heavy rainstorm.
Damage Estimate: $4,500
Coverage: Homeowner's policy with $1,000 deductible
Policy Exclusion: Gradual water damage and wear and tear

TASK:
Draft a claim partial approval letter to the policyholder explaining covered and non-covered portions.
            """,
            reference_output="""
Dear Ms. Doe,

Re: Claim Number CL-12398765, Policy Number HP-87654321

Thank you for submitting your claim regarding the water damage that occurred on June 10, 2023, at your property. We have completed our investigation of the damage to your ceiling and walls in the upstairs bathroom resulting from the roof leak during the recent heavy rainstorm.

After careful review, we have determined that your claim is partially approved under your homeowner's policy. Our investigation found that part of the damage was caused by a sudden and accidental roof leak, which is covered under your policy. However, our inspection also revealed evidence of long-term water intrusion due to gradual deterioration of roof flashing, which falls under the policy exclusion for gradual water damage and wear and tear.

Based on our assessment:
- The water damage to the bathroom ceiling and walls is covered: $3,200
- The repair of the roof leak source is not covered due to gradual wear and tear: $1,300
- Total covered amount: $3,200
- Less your deductible of $1,000
- Net payment approved: $2,200

A check for $2,200 will be mailed to your address on file within 5-7 business days. If you've signed up for direct deposit, the funds will be transferred to your account within 3 business days.

We understand this may not be the outcome you were hoping for regarding the roof repair. While your policy doesn't cover the repair of the roof itself due to the wear and tear exclusion, we strongly recommend addressing this issue promptly to prevent future damage.

If you wish to appeal this decision or provide additional information for our consideration, please contact our claims department at (555) 987-6543 within 30 days. You may also request a copy of the inspection report for your records.

Should you have any questions about your claim or the payment, please don't hesitate to contact me directly at (555) 987-6543 ext. 123 or jane.claims@abcinsurance.com.

Thank you for your business. We value you as a customer and are committed to providing fair claim settlements.

Sincerely,

Claims Representative
ABC Insurance Company
            """.strip(),
            metadata={
                "claim_type": "homeowner",
                "response_type": "partial_approval",
                "required_phrases": ["partially approved", "deductible", "exclusion", "not covered"],
                "prohibited_phrases": ["fully denied", "completely rejected"]
            }
        )
    ]

    benchmark = Benchmark(
        name="claim_response_benchmark",
        task_type="claim_response",
        description="Benchmark for evaluating insurance claim response generation",
        examples=examples,
        metrics=["relevance", "compliance", "completeness", "complexity"]
    )

    return benchmark
