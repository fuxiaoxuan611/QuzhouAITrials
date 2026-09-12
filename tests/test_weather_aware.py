import json
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from serving.management_events import FertilizerEvent, ManagementTimeline
from serving.decision_engine import DecisionEngine
from serving.forward_simulation import ForwardSimulator
from serving.operation_advice import OperationAdvice
from serving.scenario_evaluation import ManagementScenario, ScenarioEvaluator
from serving.season_calendar import SeasonCalendarResolver
from serving.weather_records import WeatherDataKind, WeatherRecord
from serving.weather_providers.base import WeatherProvider
from serving.weather_providers.timeline import TimelineWeatherDataProvider
from serving.weather_service import HistoricalForecastTimelineBuilder, WeatherService
from serving.errors import WeatherLiveQueryDayUnavailableError, WeatherTimelineInvalidError
from serving.wofost_realtime import WOFOSTRealtimeEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "CN-Maize.yaml"


def weather_record(day: date, kind: WeatherDataKind) -> WeatherRecord:
    retrieved = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return WeatherRecord(
        date=day,
        latitude=36.77,
        longitude=114.96,
        elevation=50.0,
        tmin_c=20.0,
        tmax_c=32.0,
        rain_mm=0.0,
        irrad_mj_m2_day=20.0,
        vap_hpa=18.0,
        wind_m_s=2.0,
        e0_mm=4.0,
        es0_mm=4.0,
        et0_mm=4.0,
        provider="test-weather",
        source="deterministic-test",
        data_kind=kind,
        temp_c=26.0,
        model="deterministic",
        model_run="test-run",
        retrieved_at=retrieved,
        valid_time=datetime.combine(day, datetime.min.time()),
        as_of=retrieved if kind != WeatherDataKind.HISTORICAL else None,
        timezone="Asia/Shanghai",
    )


def date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


class RecordingProvider(WeatherProvider):
    name = "test-weather"
    supports_historical = True
    supports_recent_history = False
    supports_forecast = True
    forecast_max_days = 16

    def __init__(self, *, recent=False, forecast=True):
        self.supports_recent_history = recent
        self.supports_forecast = forecast
        self.calls = []

    def _records(self, start_date, end_date, kind):
        return tuple(weather_record(day, kind) for day in date_range(start_date, end_date))

    def get_historical(self, start_date, end_date, **kwargs):
        self.calls.append(("historical", start_date, end_date))
        return self._records(start_date, end_date, WeatherDataKind.HISTORICAL)

    def get_recent_history(self, start_date, end_date, **kwargs):
        self.calls.append(("recent", start_date, end_date))
        return self._records(start_date, end_date, WeatherDataKind.OBSERVED)

    def get_forecast(self, start_date, end_date, **kwargs):
        self.calls.append(("forecast", start_date, end_date))
        return self._records(start_date, end_date, WeatherDataKind.FORECAST)


class TestWeatherServiceBoundaries(unittest.TestCase):
    campaign = date(2026, 6, 1)
    query = date(2026, 6, 10)

    def service(self, provider):
        return WeatherService(
            providers={provider.name: provider},
            default_provider=provider.name,
            forecast_default_days=2,
        )

    def test_live_archive_stops_before_query_and_forecast_owns_query_day(self):
        provider = RecordingProvider()
        result = self.service(provider).get_context(
            campaign_start=self.campaign,
            query_date=self.query,
            forecast_horizon_days=2,
            decision_mode="live",
            today=self.query,
        )
        self.assertIn(("historical", self.campaign, self.query - timedelta(days=1)), provider.calls)
        self.assertIn(("forecast", self.query, self.query + timedelta(days=2)), provider.calls)
        self.assertEqual(result["historical"]["coverage_end"], "2026-06-09")
        self.assertEqual(result["forecast"]["coverage_start"], "2026-06-10")
        self.assertEqual(result["forecast"]["records"][0]["data_kind"], "forecast")
        self.assertEqual(len({item.date for item in result["timeline"]}), 12)

    def test_live_without_safe_query_day_fails_explicitly(self):
        provider = RecordingProvider(forecast=False)
        with self.assertRaises(WeatherLiveQueryDayUnavailableError):
            self.service(provider).get_context(
                campaign_start=self.campaign,
                query_date=self.query,
                forecast_horizon_days=0,
                decision_mode="live",
                today=self.query,
            )
        self.assertIn(("historical", self.campaign, self.query - timedelta(days=1)), provider.calls)
        self.assertFalse(any(call[0] == "forecast" for call in provider.calls))

    def test_live_recent_provider_can_supply_query_day_without_archive_leakage(self):
        provider = RecordingProvider(recent=True, forecast=False)
        result = self.service(provider).get_context(
            campaign_start=self.campaign,
            query_date=self.query,
            forecast_horizon_days=0,
            decision_mode="live",
            today=self.query,
        )
        self.assertIn(("recent", self.campaign, self.query), provider.calls)
        self.assertFalse(any(call[0] == "historical" for call in provider.calls))
        self.assertEqual(result["forecast"]["records"][0]["data_kind"], "observed")

    def test_historical_replay_allows_complete_query_day_history(self):
        provider = RecordingProvider()
        result = self.service(provider).get_context(
            campaign_start=self.campaign,
            query_date=self.query,
            forecast_horizon_days=2,
            decision_mode="historical_replay",
            today=date(2026, 6, 30),
        )
        self.assertIn(("historical", self.campaign, self.query + timedelta(days=2)), provider.calls)
        self.assertFalse(any(call[0] == "forecast" for call in provider.calls))
        self.assertEqual(result["historical"]["coverage_end"], "2026-06-10")
        self.assertEqual(result["forecast"]["coverage_start"], "2026-06-11")
        self.assertTrue(
            all(record["data_kind"] == "historical" for record in result["forecast"]["records"])
        )

    def test_historical_replay_uses_archive_for_past_query_and_horizon(self):
        provider = RecordingProvider()
        result = self.service(provider).get_context(
            campaign_start=self.campaign,
            query_date=date(2026, 6, 16),
            forecast_horizon_days=7,
            decision_mode="historical_replay",
            today=date(2026, 9, 13),
        )
        self.assertIn(("historical", self.campaign, date(2026, 6, 23)), provider.calls)
        self.assertFalse(any(call[0] == "forecast" for call in provider.calls))
        self.assertEqual(result["historical"]["coverage_end"], "2026-06-16")
        self.assertEqual(result["forecast"]["coverage_start"], "2026-06-17")
        self.assertEqual(result["forecast"]["coverage_end"], "2026-06-23")
        self.assertTrue(
            all(record["data_kind"] == "historical" for record in result["forecast"]["records"])
        )

    def test_historical_replay_splits_archive_and_forecast_at_real_today(self):
        provider = RecordingProvider()
        result = self.service(provider).get_context(
            campaign_start=date(2026, 9, 10),
            query_date=date(2026, 9, 10),
            forecast_horizon_days=4,
            decision_mode="historical_replay",
            today=date(2026, 9, 12),
        )
        self.assertIn(("historical", date(2026, 9, 10), date(2026, 9, 11)), provider.calls)
        self.assertIn(("forecast", date(2026, 9, 12), date(2026, 9, 14)), provider.calls)
        self.assertEqual(
            [record.date.isoformat() for record in result["timeline"]],
            ["2026-09-10", "2026-09-11", "2026-09-12", "2026-09-13", "2026-09-14"],
        )
        self.assertEqual(
            [record["data_kind"] for record in result["forecast"]["records"]],
            ["historical", "forecast", "forecast", "forecast"],
        )

    def test_live_query_day_duplicate_is_rejected(self):
        builder = HistoricalForecastTimelineBuilder()
        query_record = weather_record(self.query, WeatherDataKind.FORECAST)
        with self.assertRaises(WeatherTimelineInvalidError):
            builder.build(
                campaign_start=self.campaign,
                query_date=self.query,
                forecast_end=self.query,
                decision_mode="live",
                historical=tuple(
                    weather_record(day, WeatherDataKind.HISTORICAL)
                    for day in date_range(self.campaign, self.query - timedelta(days=1))
                ),
                recent=(query_record,),
                forecast=(query_record,),
            )


class TestDynamicManagementEquivalence(unittest.TestCase):
    @staticmethod
    def records():
        return tuple(
            weather_record(day, WeatherDataKind.FORECAST)
            for day in date_range(date(2026, 5, 9), date(2026, 10, 15))
        )

    @staticmethod
    def snapshot(engine):
        latest = engine.env.model.get_output()[-1]
        fields = ("NO3", "NH4", "NuptakeTotal", "NLOSSCUM", "TAGP", "LAI", "WSO")
        return {field: np.asarray(latest[field], dtype=float).copy() for field in fields if field in latest}

    def test_dated_fertilizer_is_numerically_equivalent_to_action_index_two(self):
        calendar = SeasonCalendarResolver().resolve(
            "maize", "Quzhou_maize_2025_Opt", date(2026, 6, 9)
        )
        legacy = WOFOSTRealtimeEngine(
            CONFIG_PATH,
            seed=107,
            season_calendar=calendar,
            timeline_provider=TimelineWeatherDataProvider(self.records()),
        )
        dynamic = WOFOSTRealtimeEngine(
            CONFIG_PATH,
            seed=107,
            season_calendar=calendar,
            timeline_provider=TimelineWeatherDataProvider(self.records()),
            management_timeline=ManagementTimeline.from_values(
                [{"date": "2026-06-16", "n_rate_kg_ha": 20.0}], []
            ),
        )
        try:
            legacy._reset()
            legacy.env.step(2)  # CN-Maize mapping: action index 2 -> 20 kg N/ha.
            legacy_after_application = self.snapshot(legacy)
            legacy.env.step(0)
            legacy_next_timestep = self.snapshot(legacy)

            dynamic_after_application = dynamic.reconstruct(date(2026, 6, 16))
            dynamic_next_timestep = dynamic.reconstruct(date(2026, 6, 23))
            for expected, actual in (
                (legacy_after_application, dynamic_after_application["crop_state"]),
                (legacy_next_timestep, dynamic_next_timestep["crop_state"]),
            ):
                for field, expected_value in expected.items():
                    np.testing.assert_allclose(
                        np.asarray(actual[field], dtype=float), expected_value, rtol=1e-8, atol=1e-8
                    )
            self.assertEqual(
                ManagementTimeline.from_values(
                    [{"date": "2026-06-16", "n_rate_kg_ha": 20.0}], []
                ).fertilizers[0].to_pcse_timed_event()["events_table"][0][date(2026, 6, 16)]["amount"],
                20.0,
            )
        finally:
            legacy.close()
            dynamic.close()


class TestDynamicContracts(unittest.TestCase):
    def test_dynamic_season_defaults_and_harvest_override(self):
        resolver = SeasonCalendarResolver()
        calendar = resolver.resolve("maize", "Quzhou_maize_2025_Opt", date(2026, 6, 9))
        self.assertEqual(calendar.campaign_start_date, date(2026, 5, 9))
        self.assertEqual(calendar.crop_end_date, date(2026, 10, 15))
        overridden = resolver.resolve(
            "maize", "Quzhou_maize_2025_Opt", date(2026, 6, 9), date(2026, 10, 1)
        )
        self.assertEqual(overridden.crop_end_date, date(2026, 10, 1))

    def test_management_conversion_is_json_safe(self):
        event = ManagementTimeline.from_values(
            [{"date": "2026-06-16", "n_rate_kg_ha": 20.0}],
            [{"date": "2026-07-01", "amount_mm": 20.0, "efficiency": 0.8}],
        )
        payload = event.to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertEqual(payload["fertilization_history"][0]["n_recovery"], 0.7)
        self.assertNotIn("N_recovery", event.fertilizers[0].to_pcse_timed_event()["events_table"][0][date(2026, 6, 16)])
        irrigation = event.irrigations[0].to_pcse_timed_event()
        self.assertAlmostEqual(irrigation["events_table"][0][date(2026, 7, 1)]["amount"], 2.0)

    def test_effective_irrigation_amount_is_not_double_reduced(self):
        timeline = ManagementTimeline.from_values(
            [],
            [{"date": "2026-07-01", "amount_mm": 20.0, "efficiency": 0.8, "amount_basis": "effective"}],
        )
        event = timeline.irrigations[0]
        self.assertAlmostEqual(event.pcse_amount_cm, 2.5)
        params = event.to_pcse_timed_event()["events_table"][0][date(2026, 7, 1)]
        self.assertAlmostEqual(params["amount"] * params["efficiency"], 2.0)

    def test_operation_advice_preserves_rl_rate_when_timing_changes(self):
        advice = OperationAdvice().build(
            rl_recommendation={"n_rate_kg_ha": 20.0},
            weather_risk=[{"risk_type": "heavy_rain", "severity": "high"}],
        )
        self.assertEqual(advice["recommended_n_rate_kg_ha"], 20.0)
        self.assertEqual(advice["timing_status"], "delay")

    def test_scenario_order_isolation_with_deterministic_simulator(self):
        class FakeSimulator:
            def simulate(self, **kwargs):
                management = kwargs["management"]
                total_n = sum(item.n_rate_kg_ha for item in management.fertilizers)
                return type("Result", (), {"to_dict": lambda _self: {"state_at_horizon": {"total_n": total_n}}})()

        scenarios = (
            ManagementScenario("A"),
            ManagementScenario("B", fertilizer_events=(
                FertilizerEvent(date(2026, 9, 5), 20),
            )),
            ManagementScenario("C", fertilizer_events=(
                FertilizerEvent(date(2026, 9, 8), 20),
            )),
        )
        calendar = SeasonCalendarResolver().resolve("maize", "Quzhou_maize_2025_Opt", date(2026, 6, 9))
        historical = ManagementTimeline.from_values([], [])
        evaluator = ScenarioEvaluator(FakeSimulator())
        forward = evaluator.evaluate(
            scenarios=scenarios,
            calendar=calendar,
            weather_records=(),
            historical_management=historical,
            query_date=date(2026, 9, 3),
            horizon_date=date(2026, 9, 10),
        )
        reverse = evaluator.evaluate(
            scenarios=tuple(reversed(scenarios)),
            calendar=calendar,
            weather_records=(),
            historical_management=historical,
            query_date=date(2026, 9, 3),
            horizon_date=date(2026, 9, 10),
        )
        self.assertEqual(
            {item["scenario_id"]: item["state_at_horizon"] for item in forward},
            {item["scenario_id"]: item["state_at_horizon"] for item in reverse},
        )

    def test_real_scenarios_advance_apply_events_and_remain_isolated(self):
        scenarios = (
            ManagementScenario("A"),
            ManagementScenario(
                "B",
                fertilizer_events=(FertilizerEvent(date(2026, 9, 5), 20),),
            ),
            ManagementScenario(
                "C",
                fertilizer_events=(FertilizerEvent(date(2026, 9, 8), 20),),
            ),
        )
        calendar = SeasonCalendarResolver().resolve(
            "maize", "Quzhou_maize_2025_Opt", date(2026, 6, 9)
        )
        records = tuple(
            weather_record(day, WeatherDataKind.FORECAST)
            for day in date_range(calendar.campaign_start_date, date(2026, 9, 10))
        )
        historical = ManagementTimeline.from_values(
            [{"date": "2026-08-25", "n_rate_kg_ha": 30.0}], []
        )
        evaluator = ScenarioEvaluator(ForwardSimulator(CONFIG_PATH))

        def evaluate(values):
            return {
                item["scenario_id"]: item
                for item in evaluator.evaluate(
                    scenarios=values,
                    calendar=calendar,
                    weather_records=records,
                    historical_management=historical,
                    query_date=date(2026, 9, 3),
                    horizon_date=date(2026, 9, 10),
                )
            }

        forward = evaluate(scenarios)
        reverse = evaluate(tuple(reversed(scenarios)))
        self.assertEqual(
            {key: value["state_at_horizon"] for key, value in forward.items()},
            {key: value["state_at_horizon"] for key, value in reverse.items()},
        )
        self.assertEqual(
            {json.dumps(value["state_at_query"], sort_keys=True) for value in forward.values()},
            {json.dumps(forward["A"]["state_at_query"], sort_keys=True)},
        )
        for value in forward.values():
            self.assertNotEqual(value["state_at_query"], value["state_at_horizon"])

        def mineral_n(state):
            return sum(state["NO3"]) + sum(state["NH4"])

        self.assertNotAlmostEqual(
            mineral_n(forward["A"]["state_at_horizon"]),
            mineral_n(forward["B"]["state_at_horizon"]),
        )
        self.assertNotAlmostEqual(
            mineral_n(forward["A"]["state_at_horizon"]),
            mineral_n(forward["C"]["state_at_horizon"]),
        )


class TestDynamicAcceptance(unittest.TestCase):
    def test_2026_off_boundary_request_projects_next_decision(self):
        run = PROJECT_ROOT / "tensorboard_logs" / "WOFOST_maize_experiments" / "CN-Maize-Seed-107-nsteps-2208-LagPPO-POT-run_1"
        provider = RecordingProvider()
        provider.name = "openmeteo"
        service = WeatherService(
            providers={"openmeteo": provider},
            default_provider="openmeteo",
            forecast_default_days=7,
        )
        engine = DecisionEngine(
            config_path=CONFIG_PATH,
            model_path=run / "latest-model.zip",
            env_stats_path=run / "latest-env.pkl",
            device="cpu",
            policy_validation_status="engineering_only",
            weather_service=service,
        )
        try:
            result = engine.decide({
                "schema_version": "1.0",
                "request": {"request_id": "acceptance-2026"},
                "location": {"latitude": 36.77, "longitude": 114.96},
                "crop": {
                    "name": "maize",
                    "cultivar": "Quzhou_maize_2025_Opt",
                    "sowing_date": "2026-06-09",
                },
                "query_date": "2026-09-03",
                "management": {
                    "fertilization_history_complete": True,
                    "fertilization_history": [
                        {"date": "2026-08-25", "n_rate_kg_ha": 30.0},
                    ],
                    "irrigation_history_complete": True,
                    "irrigation_history": [],
                },
                "observations": None,
                "weather": {
                    "provider": "openmeteo",
                    "use_external_provider": True,
                },
                "decision_context": {"decision_mode": "historical_replay"},
            })
            self.assertFalse(result["decision_due"])
            self.assertIsNone(result["recommendation"])
            self.assertEqual(result["previous_decision_date"], "2026-09-01")
            self.assertEqual(result["next_decision_date"], "2026-09-08")
            self.assertTrue(result["projected_next_decision"]["projected"])
            self.assertTrue(result["projected_next_decision"]["uses_forecast"])
            self.assertEqual(result["projected_next_decision"]["decision_date"], "2026-09-08")
            self.assertEqual(
                result["projected_next_decision"]["action_application_date"],
                "2026-09-15",
            )
            self.assertEqual(result["model_metadata"]["validation_status"], "engineering_only")
            self.assertFalse(result["model_metadata"]["validated_for_agronomic_recommendation"])
            self.assertEqual(result["weather_context"]["forecast"]["coverage_start"], "2026-09-04")
            self.assertEqual(result["weather_context"]["forecast"]["coverage_end"], "2026-09-17")
            self.assertEqual(result["forecast"]["horizon_days"], 7)
            self.assertEqual(result["forecast"]["horizon_end"], "2026-09-10")
            self.assertTrue(result["forecast"]["uses_forecast"])
            self.assertTrue(result["forecast"]["state_estimated"])
            self.assertEqual(result["forecast"]["scenario_horizon_end"], "2026-09-17")

            scenarios = {item["scenario_id"]: item for item in result["scenario_evaluation"]}
            self.assertEqual(set(scenarios), {"no_additional_n", "rl_policy_rate"})
            self.assertEqual(
                scenarios["rl_policy_rate"]["fertilizer_events"][0]["date"],
                "2026-09-15",
            )
            for scenario in scenarios.values():
                self.assertEqual(scenario["horizon_date"], "2026-09-17")
                self.assertNotEqual(scenario["state_at_query"], scenario["state_at_horizon"])
            self.assertNotEqual(
                scenarios["no_additional_n"]["state_at_horizon"]["NO3"],
                scenarios["rl_policy_rate"]["state_at_horizon"]["NO3"],
            )
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
