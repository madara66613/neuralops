from neuralops.modeling.error_analysis import _distribution, _summarize_cases


def test_distribution_handles_empty_and_observed_values() -> None:
    assert _distribution([])["median"] is None
    observed = _distribution([0.1, 0.9, 0.5])
    assert observed == {
        "count": 3,
        "minimum": 0.1,
        "median": 0.5,
        "mean": 0.5,
        "maximum": 0.9,
    }


def test_case_summary_counts_events_and_diagnostics() -> None:
    cases = [
        {
            "events": ["E1", "E2", "E2"],
            "anomaly_probability": 0.8,
            "sequence_length": 3,
            "unique_event_count": 2,
            "manual_review": True,
            "truncated": False,
            "unknown_event_count": 1,
        },
        {
            "events": ["E2", "E3"],
            "anomaly_probability": 0.9,
            "sequence_length": 2,
            "unique_event_count": 2,
            "manual_review": False,
            "truncated": True,
            "unknown_event_count": 0,
        },
    ]
    summary = _summarize_cases(cases)
    assert summary["count"] == 2
    assert summary["manual_review_count"] == 1
    assert summary["truncated_count"] == 1
    assert summary["unknown_event_count"] == 1
    assert summary["common_events"][0] == {
        "event": "E2",
        "occurrences": 3,
        "case_count": 2,
        "case_share": 1.0,
    }
