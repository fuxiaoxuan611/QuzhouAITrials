"""Dynamic, auditable crop-season calendar resolution.

The 2025 YAML calendar remains the reproducibility baseline.  This module is
used when a request supplies a different season and creates the equivalent
agromanagement calendar in memory; it never writes a new YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any


class DynamicCalendarError(ValueError):
    """Raised when a dynamic crop calendar cannot be resolved."""

    code = "DYNAMIC_CALENDAR_INVALID"


@dataclass(frozen=True)
class SeasonCalendar:
    crop_name: str
    cultivar: str
    sowing_date: date
    campaign_start_date: date
    crop_start_date: date
    crop_end_date: date
    crop_start_type: str
    crop_end_type: str
    max_duration: int
    assumptions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_name": self.crop_name,
            "cultivar": self.cultivar,
            "sowing_date": self.sowing_date.isoformat(),
            "campaign_start_date": self.campaign_start_date.isoformat(),
            "crop_start_date": self.crop_start_date.isoformat(),
            "crop_end_date": self.crop_end_date.isoformat(),
            "crop_start_type": self.crop_start_type,
            "crop_end_type": self.crop_end_type,
            "max_duration": self.max_duration,
            "assumptions": dict(self.assumptions),
        }

    def agromanagement_structure(self, management_events: Any = None) -> list[dict[str, Any]]:
        """Return the PCSE agromanagement structure for this season.

        ``management_events`` may be a ``ManagementTimeline`` or an iterable
        of event objects.  Conversion is intentionally duck-typed so this
        module does not create a dependency cycle with management_events.
        """

        timed_events: list[dict[str, Any]] = []
        if management_events is not None:
            to_pcse = getattr(management_events, "to_pcse_timed_events", None)
            if callable(to_pcse):
                timed_events = to_pcse()
            else:
                for event in management_events:
                    converter = getattr(event, "to_pcse_timed_event", None)
                    if callable(converter):
                        timed_events.append(converter())
        return [
            {
                self.campaign_start_date: {
                    "CropCalendar": {
                        "crop_name": self.crop_name,
                        "variety_name": self.cultivar,
                        "crop_start_date": self.crop_start_date,
                        "crop_start_type": self.crop_start_type,
                        "crop_end_date": self.crop_end_date,
                        "crop_end_type": self.crop_end_type,
                        "max_duration": self.max_duration,
                    },
                    "TimedEvents": timed_events or None,
                    "StateEvents": None,
                }
            }
        ]


class SeasonCalendarResolver:
    """Resolve request dates while making every calendar assumption explicit."""

    def __init__(
        self,
        *,
        pre_sowing_offset_days: int = 31,
        crop_duration_days: int = 128,
        max_duration: int = 400,
        assumption_name: str = "quzhou_2025_baseline_assumption",
    ) -> None:
        if pre_sowing_offset_days < 0 or crop_duration_days <= 0 or max_duration <= 0:
            raise DynamicCalendarError("calendar defaults must be non-negative/positive")
        if crop_duration_days > max_duration:
            raise DynamicCalendarError("crop_duration_days cannot exceed max_duration")
        self.pre_sowing_offset_days = int(pre_sowing_offset_days)
        self.crop_duration_days = int(crop_duration_days)
        self.max_duration = int(max_duration)
        self.assumption_name = str(assumption_name)

    def resolve(
        self,
        crop_name: str,
        cultivar: str,
        sowing_date: date,
        expected_harvest_date: date | None = None,
        pre_sowing_offset_days: int | None = None,
        crop_duration_days: int | None = None,
    ) -> SeasonCalendar:
        if not isinstance(crop_name, str) or not crop_name.strip():
            raise DynamicCalendarError("crop_name must be a non-empty string")
        if not isinstance(cultivar, str) or not cultivar.strip():
            raise DynamicCalendarError("cultivar must be a non-empty string")
        if not isinstance(sowing_date, date):
            raise DynamicCalendarError("sowing_date must be a date")
        offset = self.pre_sowing_offset_days if pre_sowing_offset_days is None else int(pre_sowing_offset_days)
        duration = self.crop_duration_days if crop_duration_days is None else int(crop_duration_days)
        if offset < 0:
            raise DynamicCalendarError("pre_sowing_offset_days must be >= 0")
        if duration <= 0 or duration > self.max_duration:
            raise DynamicCalendarError("crop_duration_days must be in (0, max_duration]")
        if expected_harvest_date is not None:
            if not isinstance(expected_harvest_date, date) or expected_harvest_date <= sowing_date:
                raise DynamicCalendarError("expected_harvest_date must be after sowing_date")
            crop_end = expected_harvest_date
            end_source = "request.expected_harvest_date"
        else:
            crop_end = sowing_date + timedelta(days=duration)
            end_source = "configured_crop_duration_days"
        if (crop_end - sowing_date).days > self.max_duration:
            raise DynamicCalendarError("crop end exceeds configured max_duration")
        return SeasonCalendar(
            crop_name=crop_name,
            cultivar=cultivar,
            sowing_date=sowing_date,
            campaign_start_date=sowing_date - timedelta(days=offset),
            crop_start_date=sowing_date,
            crop_end_date=crop_end,
            crop_start_type="sowing",
            crop_end_type="harvest",
            max_duration=self.max_duration,
            assumptions={
                "name": self.assumption_name,
                "pre_sowing_offset_days": offset,
                "crop_duration_days": duration,
                "crop_duration_source": end_source,
                "max_duration": self.max_duration,
            },
        )


__all__ = ["DynamicCalendarError", "SeasonCalendar", "SeasonCalendarResolver"]
