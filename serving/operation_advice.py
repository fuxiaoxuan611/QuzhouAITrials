"""Structured operation advice with RL-owned nitrogen quantities."""

from __future__ import annotations

from datetime import date
from typing import Any, Iterable


class OperationAdvice:
    """Build timing advice without inventing an optimized N rate."""

    def build(
        self,
        *,
        rl_recommendation: dict[str, Any] | None = None,
        projected_recommendation: dict[str, Any] | None = None,
        weather_risk: Iterable[dict[str, Any]] = (),
        next_evaluation_date: date | None = None,
    ) -> dict[str, Any]:
        recommendation = rl_recommendation or projected_recommendation
        risks = tuple(weather_risk)
        high_rain = any(item.get("risk_type") in {"heavy_rain", "consecutive_rain"} and item.get("severity") in {"high", "severe"} for item in risks)
        if recommendation is None:
            return {
                "status": "wait_monitor",
                "recommended_n_rate_kg_ha": None,
                "timing_status": "monitor",
                "preferred_window": None,
                "reason_codes": ["NO_CURRENT_OR_PROJECTED_RL_RATE"],
                "recommendation_source": None,
                "timing_source": None,
                "next_evaluation_date": next_evaluation_date.isoformat() if next_evaluation_date else None,
            }
        if high_rain:
            timing = "delay"
            reasons = ["WEATHER_HEAVY_RAIN_RISK"]
        else:
            timing = "execute_or_monitor"
            reasons = []
        return {
            "status": "actionable" if not high_rain else "timing_adjusted",
            "recommended_n_rate_kg_ha": float(recommendation["n_rate_kg_ha"]),
            "timing_status": timing,
            "preferred_window": None,
            "reason_codes": reasons,
            "recommendation_source": "rl_policy",
            "timing_source": "weather_risk" if high_rain else None,
            "next_evaluation_date": next_evaluation_date.isoformat() if next_evaluation_date else None,
        }


__all__ = ["OperationAdvice"]
