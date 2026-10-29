"""NOVA Market Regime Intelligence subsystem (Person 2).

Provides canonical, deterministic, causal market-regime classification:
- Kaufman Efficiency Ratio (KER)
- RiskMetrics EWMA volatility
- Directional EMA filtering
- Point-in-time streaming and warm-start execution
- Interface Contract B (ModelPrediction) compliance
"""

from nova.regime.config import MarketRegimeConfig
from nova.regime.transparent_detector import (
    RegimeLabel,
    TransparentRegimeDetector,
)

__all__ = [
    "MarketRegimeConfig",
    "RegimeLabel",
    "TransparentRegimeDetector",
]

