"""Authoritative historical/forecast weather orchestration layer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from .errors import (
    ForecastHorizonInsufficientError,
    WeatherDataGapError,
    WeatherLiveQueryDayUnavailableError,
    WeatherProviderError,
    WeatherTimelineInvalidError,
)
from .weather_records import WeatherDataKind, WeatherRecord
from .weather_providers.base import WeatherProvider


def _as_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _coerce_records(values: Iterable[WeatherRecord | Mapping[str, Any]] | None) -> tuple[WeatherRecord, ...]:
    if values is None:
        return ()
    result = []
    for value in values:
        if isinstance(value, WeatherRecord):
            result.append(value)
        elif isinstance(value, Mapping):
            result.append(WeatherRecord.from_mapping(value))
        else:
            raise WeatherTimelineInvalidError("weather records must be WeatherRecord or object mappings")
    return tuple(result)


class HistoricalForecastTimelineBuilder:
    """Select and validate one continuous daily history/forecast timeline."""

    def build(
        self,
        *,
        campaign_start: date | str,
        query_date: date | str,
        forecast_end: date | str,
        decision_mode: str = "auto",
        historical: Iterable[WeatherRecord | Mapping[str, Any]] = (),
        recent: Iterable[WeatherRecord | Mapping[str, Any]] = (),
        forecast: Iterable[WeatherRecord | Mapping[str, Any]] = (),
        today: date | None = None,
    ) -> tuple[WeatherRecord, ...]:
        campaign = _as_date(campaign_start)
        query = _as_date(query_date)
        end = _as_date(forecast_end)
        if campaign > query or end < query:
            raise WeatherTimelineInvalidError("weather timeline date range is invalid")
        if decision_mode not in {"auto", "historical_replay", "live", "simulation"}:
            raise WeatherTimelineInvalidError("unsupported decision_mode")
        historical_values = _coerce_records(historical)
        recent_values = _coerce_records(recent)
        forecast_values = _coerce_records(forecast)
        if decision_mode == "auto":
            local_today = today or datetime.now().date()
            mode = "live" if query == local_today else "historical_replay"
        else:
            mode = decision_mode

        selected: list[WeatherRecord] = []
        if mode in {"historical_replay", "simulation"}:
            selected.extend(item for item in historical_values if campaign <= item.date <= query)
            if mode == "simulation":
                selected.extend(item for item in recent_values if campaign <= item.date <= query)
            selected.extend(item for item in forecast_values if query < item.date <= end)
        else:  # live: the query date is not allowed to leak into historical data.
            selected.extend(item for item in historical_values if campaign <= item.date < query)
            selected.extend(item for item in recent_values if item.date == query)
            selected.extend(item for item in forecast_values if query <= item.date <= end)

        by_date: dict[date, WeatherRecord] = {}
        for record in selected:
            if record.date in by_date:
                raise WeatherTimelineInvalidError(f"duplicate weather date: {record.date}")
            by_date[record.date] = record
        expected = campaign
        while expected <= end:
            if expected not in by_date:
                raise WeatherDataGapError(f"missing weather date: {expected}")
            expected += timedelta(days=1)
        result = tuple(by_date[item] for item in sorted(by_date))
        timezones = {item.timezone for item in result}
        if len(timezones) != 1:
            raise WeatherTimelineInvalidError("weather timeline has inconsistent timezones")
        if any(item.date < campaign or item.date > end for item in result):
            raise WeatherTimelineInvalidError("weather timeline contains an out-of-range record")
        return result


@dataclass(frozen=True)
class WeatherService:
    """Provider registry, provenance, cache, and timeline policy."""

    providers: Mapping[str, WeatherProvider]
    default_provider: str = "openmeteo"
    default_timezone: str = "Asia/Shanghai"
    default_latitude: float = 36.77
    default_longitude: float = 114.96
    default_elevation: float = 0.0
    forecast_default_days: int = 7
    allowed_location_tolerance: float = 2.0
    fallback_provider_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "_cache_metadata", {})
        object.__setattr__(self, "timeline_builder", HistoricalForecastTimelineBuilder())

    @classmethod
    def default(cls, *, timezone_name: str = "Asia/Shanghai") -> "WeatherService":
        from .weather_providers.openmeteo import OpenMeteoWeatherProvider

        return cls(
            providers={"openmeteo": OpenMeteoWeatherProvider(timezone_name=timezone_name)},
            default_timezone=timezone_name,
        )

    def cache_key(
        self,
        *,
        provider: str,
        source: str,
        model: str | None,
        latitude: float,
        longitude: float,
        elevation: float,
        timezone_name: str,
        start_date: date,
        end_date: date,
        data_kind: str,
        model_run: str | None = None,
        as_of: datetime | None = None,
        et_model: str = "PM",
        requested_variables: Iterable[str] = (),
        unit_conversion_version: str = "weather-record-v1",
    ) -> str:
        payload = {
            "provider": provider,
            "source": source,
            "model": model,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "elevation": float(elevation),
            "timezone": timezone_name,
            "start_date": _as_date(start_date).isoformat(),
            "end_date": _as_date(end_date).isoformat(),
            "data_kind": data_kind,
            "model_run": model_run,
            "as_of": as_of.isoformat() if as_of else None,
            "ETmodel": et_model,
            "requested_variables": sorted(str(item) for item in requested_variables),
            "unit_conversion_version": unit_conversion_version,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _cache_records(self, records: tuple[WeatherRecord, ...], key: str) -> tuple[WeatherRecord, ...]:
        if not records or any(not isinstance(item, WeatherRecord) for item in records):
            raise WeatherTimelineInvalidError("cached weather payload is corrupt")
        self._cache[key] = records
        self._cache_metadata[key] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "retrieved_at": max(
                (item.retrieved_at.isoformat() for item in records if item.retrieved_at),
                default=None,
            ),
            "coverage_start": min(item.date for item in records).isoformat(),
            "coverage_end": max(item.date for item in records).isoformat(),
            "provider": records[0].provider,
            "model": records[0].model,
            "model_run": records[0].model_run,
            "as_of": records[0].as_of.isoformat() if records[0].as_of else None,
        }
        return records

    def _fetch(
        self,
        provider: WeatherProvider,
        kind: str,
        start: date,
        end: date,
        *,
        latitude: float,
        longitude: float,
        elevation: float,
        timezone_name: str,
        as_of: datetime | None,
    ) -> tuple[WeatherRecord, ...]:
        if end < start:
            return ()
        key = self.cache_key(
            provider=provider.name,
            source=provider.name,
            model=getattr(provider, "model", None),
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            timezone_name=timezone_name,
            start_date=start,
            end_date=end,
            data_kind=kind,
            as_of=as_of,
            requested_variables=("tmin_c", "tmax_c", "rain_mm", "irrad_mj_m2_day", "et0_mm"),
        )
        if key in self._cache:
            cached = self._cache[key]
            if not isinstance(cached, tuple) or any(not isinstance(item, WeatherRecord) for item in cached):
                del self._cache[key]
            else:
                expected_dates = {start + timedelta(days=index) for index in range((end - start).days + 1)}
                if {item.date for item in cached} == expected_dates:
                    return cached
                # An incomplete cache entry is not authoritative.  Drop it so
                # the provider is asked for a complete range again.
                del self._cache[key]
                self._cache_metadata.pop(key, None)
        try:
            if kind == "historical":
                records = provider.get_historical(start, end, latitude=latitude, longitude=longitude, elevation=elevation, timezone=timezone_name, as_of=as_of)
            elif kind == "recent":
                records = provider.get_recent_history(start, end, latitude=latitude, longitude=longitude, elevation=elevation, timezone=timezone_name, as_of=as_of)
            elif kind == "forecast":
                records = provider.get_forecast(start, end, latitude=latitude, longitude=longitude, elevation=elevation, timezone=timezone_name, as_of=as_of)
            else:
                raise WeatherTimelineInvalidError(f"unsupported fetch kind: {kind}")
            return self._cache_records(_coerce_records(records), key)
        except (WeatherTimelineInvalidError, WeatherDataGapError):
            raise
        except Exception as exc:
            if isinstance(exc, WeatherProviderError):
                raise
            raise WeatherProviderError("weather provider request failed") from exc

    def get_context(
        self,
        *,
        campaign_start: date | str,
        query_date: date | str,
        forecast_horizon_days: int | None = None,
        decision_mode: str = "auto",
        provider_name: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        elevation: float | None = None,
        timezone_name: str | None = None,
        historical: Iterable[WeatherRecord | Mapping[str, Any]] | None = None,
        recent: Iterable[WeatherRecord | Mapping[str, Any]] | None = None,
        forecast: Iterable[WeatherRecord | Mapping[str, Any]] | None = None,
        as_of: datetime | None = None,
        today: date | None = None,
    ) -> dict[str, Any]:
        campaign = _as_date(campaign_start)
        query = _as_date(query_date)
        horizon = self.forecast_default_days if forecast_horizon_days is None else int(forecast_horizon_days)
        if horizon < 0:
            raise ForecastHorizonInsufficientError("forecast_horizon_days must be >= 0")
        lat = self.default_latitude if latitude is None else float(latitude)
        lon = self.default_longitude if longitude is None else float(longitude)
        elev = self.default_elevation if elevation is None else float(elevation)
        tz_name = timezone_name or self.default_timezone
        if abs(lat - self.default_latitude) > self.allowed_location_tolerance or abs(lon - self.default_longitude) > self.allowed_location_tolerance:
            raise WeatherProviderError("requested location is outside the configured Quzhou study area")
        selected_name = provider_name or self.default_provider
        if (
            selected_name not in self.providers
            and not any(name in self.providers for name in self.fallback_provider_names)
            and not (historical is not None or forecast is not None)
        ):
            from .errors import WeatherProviderNotConfiguredError
            raise WeatherProviderNotConfiguredError(f"weather provider is not configured: {selected_name}")
        provider = self.providers.get(selected_name)
        mode = decision_mode
        reference_today = today or datetime.now().date()
        if mode == "auto":
            mode = "live" if query == reference_today else "historical_replay"
        end = query + timedelta(days=horizon)
        warnings: list[str] = []
        actual_provider_names: list[str] = []

        def fetch_with_fallback(kind: str, start: date, finish: date) -> tuple[WeatherRecord, ...]:
            last_error: BaseException | None = None
            candidates = (selected_name, *self.fallback_provider_names)
            capability = {
                "historical": "supports_historical",
                "recent": "supports_recent_history",
                "forecast": "supports_forecast",
            }.get(kind)
            for candidate_name in candidates:
                candidate = self.providers.get(candidate_name)
                if candidate is None:
                    continue
                if capability is not None and not getattr(candidate, capability, False):
                    continue
                if kind == "forecast" and candidate.forecast_max_days is not None:
                    requested_days = (finish - start).days + 1
                    if requested_days > int(candidate.forecast_max_days):
                        last_error = ForecastHorizonInsufficientError(
                            "requested forecast horizon exceeds provider capability",
                            details={"provider": candidate.name, "max_days": candidate.forecast_max_days},
                        )
                        continue
                try:
                    values = self._fetch(
                        candidate,
                        kind,
                        start,
                        finish,
                        latitude=lat,
                        longitude=lon,
                        elevation=elev,
                        timezone_name=tz_name,
                        as_of=as_of,
                    )
                    actual_provider_names.append(candidate_name)
                    if candidate_name != selected_name:
                        warnings.append(f"weather_provider_fallback:{candidate_name}")
                    return values
                except WeatherProviderError as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
            from .errors import WeatherProviderNotConfiguredError
            raise WeatherProviderNotConfiguredError(f"weather provider is not configured: {selected_name}")

        # Provider selection is based on the real current date, not on whether
        # a date is future relative to a simulated historical query.  A
        # historical replay may therefore place archive records after its
        # query date into the timeline's future bucket without relabelling
        # them as forecasts.
        query_day_records: tuple[WeatherRecord, ...] = ()
        if historical is None:
            has_recent_provider = any(
                getattr(self.providers.get(name), "supports_recent_history", False)
                for name in (selected_name, *self.fallback_provider_names)
            )
            if mode == "live" and has_recent_provider:
                recent_values_all = fetch_with_fallback("recent", campaign, query)
                historical_source_values = tuple(
                    item for item in recent_values_all if item.date < query
                )
                query_day_records = tuple(
                    item for item in recent_values_all if item.date == query
                )
            else:
                historical_end = min(end, reference_today - timedelta(days=1))
                historical_source_values = (
                    fetch_with_fallback("historical", campaign, historical_end)
                    if historical_end >= campaign
                    and any(
                        getattr(self.providers.get(name), "supports_historical", False)
                        for name in (selected_name, *self.fallback_provider_names)
                    )
                    else ()
                )
        else:
            historical_source_values = _coerce_records(historical)
        if recent is None:
            recent_values = query_day_records
        else:
            recent_values = _coerce_records(recent)
        if forecast is None:
            has_forecast_provider = any(
                getattr(self.providers.get(name), "supports_forecast", False)
                for name in (selected_name, *self.fallback_provider_names)
            )
            forecast_start = max(campaign, reference_today)
            forecast_source_values = (
                fetch_with_fallback("forecast", forecast_start, end)
                if has_forecast_provider and end >= forecast_start
                else ()
            )
        else:
            forecast_source_values = _coerce_records(forecast)

        source_values = (*historical_source_values, *forecast_source_values)
        if mode in {"historical_replay", "simulation"}:
            historical_values = tuple(
                item for item in source_values if campaign <= item.date <= query
            )
            forecast_values = tuple(
                item for item in source_values if query < item.date <= end
            )
        else:
            historical_values = tuple(
                item for item in source_values if campaign <= item.date < query
            )
            forecast_values = tuple(
                item for item in source_values if query <= item.date <= end
            )
        if mode == "live":
            query_candidates = tuple(
                item for item in (*recent_values, *forecast_values) if item.date == query
            )
            if any(item.data_kind == WeatherDataKind.HISTORICAL for item in query_candidates):
                raise WeatherLiveQueryDayUnavailableError(
                    "live query-day weather cannot be supplied as completed historical data"
                )
            if not query_candidates:
                raise WeatherLiveQueryDayUnavailableError(
                    "live query-day weather requires a recent, nowcast, observed, or forecast record"
                )
        timeline = self.timeline_builder.build(
            campaign_start=campaign,
            query_date=query,
            forecast_end=end,
            decision_mode=mode,
            historical=historical_values,
            recent=recent_values,
            forecast=forecast_values,
            today=reference_today,
        )
        historical_selected = tuple(item for item in timeline if item.date <= query and item.data_kind in {WeatherDataKind.HISTORICAL, WeatherDataKind.OBSERVED})
        forecast_selected = tuple(item for item in timeline if item.date > query or (mode == "live" and item.date == query))
        state_estimated = mode == "live" and any(item.date == query and item.data_kind != WeatherDataKind.OBSERVED for item in timeline)
        if mode == "live":
            warnings.append("LIVE_QUERY_DAY_STATE_ESTIMATED")
        provenance = {
            "provider": actual_provider_names[-1] if actual_provider_names else selected_name,
            "source": sorted({item.source for item in timeline}),
            "models": sorted({item.model for item in timeline if item.model}),
            "model_runs": sorted({item.model_run for item in timeline if item.model_run}),
            "retrieved_at": sorted({item.retrieved_at.isoformat() for item in timeline if item.retrieved_at}),
            "valid_times": sorted({item.valid_time.isoformat() for item in timeline if item.valid_time}),
            "record_as_of": sorted({item.as_of.isoformat() for item in timeline if item.as_of}),
            "as_of": as_of.isoformat() if as_of else None,
        }
        record_as_of = sorted(item.as_of for item in timeline if item.as_of)
        return {
            "status": "ok",
            "location": {"latitude": lat, "longitude": lon, "elevation": elev},
            "timezone": tz_name,
            "decision_mode": mode,
            "campaign_start": campaign.isoformat(),
            "query_date": query.isoformat(),
            "forecast_horizon_days": horizon,
            "history_boundary": (query - timedelta(days=1)).isoformat() if mode == "live" else query.isoformat(),
            "forecast_boundary": query.isoformat() if mode == "live" else (query + timedelta(days=1)).isoformat(),
            "historical": {
                "coverage_start": historical_selected[0].date.isoformat() if historical_selected else None,
                "coverage_end": historical_selected[-1].date.isoformat() if historical_selected else None,
                "records": [item.to_dict() for item in historical_selected],
            },
            "forecast": {
                "coverage_start": forecast_selected[0].date.isoformat() if forecast_selected else None,
                "coverage_end": forecast_selected[-1].date.isoformat() if forecast_selected else None,
                "records": [item.to_dict() for item in forecast_selected],
            },
            "provenance": provenance,
            "state_estimated": state_estimated,
            "state_time_semantics": "forecast_assisted_end_of_day" if state_estimated else "end_of_day_actual_or_reanalysis",
            "as_of": (
                as_of.isoformat()
                if as_of
                else (max(record_as_of).isoformat() if record_as_of and mode == "live" else None)
            ),
            "warnings": warnings,
            "timeline": timeline,
        }

    build_context = get_context


__all__ = ["HistoricalForecastTimelineBuilder", "WeatherService"]
