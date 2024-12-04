from enum import Enum
from dataclasses import dataclass

class ModelTier(Enum):
    ANALYSIS = "analysis"
    RESPONSE = "response"
    EMBEDDING = "embedding"

@dataclass
class ModelConfig:
    name: str
    tokens_per_second: float
    cost_per_1k_tokens: float
    max_context_tokens: int
    latency_ms: float
    quality_score: float

MODEL_CONFIGS = {
    "gpt-4-turbo-preview": ModelConfig("gpt-4-turbo-preview", 30, 0.0015, 16385, 500, 0.85),
    "gpt-4o": ModelConfig("gpt-4o", 12, 0.03, 3072, 2000, 1.0),
    "text-embedding-3-large": ModelConfig("text-embedding-3-large", 100, 0.0004, 3072, 100, 0.90)
}
