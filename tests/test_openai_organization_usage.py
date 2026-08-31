"""Tests of the OpenAI-compatible organization usage and costs endpoints.

Every test drives the routes over HTTP against a stubbed CloudWatch client, so
the whole surface -- shape, refusals, pagination, caching and authorization --
runs offline and costs nothing. The response shapes are asserted against the
installed ``openai`` SDK's own models, which is what a client written for the
Administration API parses with.

The one thing a stub cannot prove is the shape of the labels the read layer
parses the dimension values out of, because CloudWatch resolves them. Measured
against the real service (``eu-west-3``, 2026-08-27): a ``Label`` of
``${PROP('Dim.Model')}|${PROP('MetricName')}`` came back as
``amazon.nova-lite-v1:0|InputTokens``, one result per matched series, each
carrying the ``Id`` of the query that produced it and arriving in no
particular order. Four ``Dim.`` placeholders in one label resolve the same
way. The stubs below reproduce that format.

Ref: https://platform.openai.com/docs/api-reference/usage
Ref: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/graph-dynamic-labels.html
Ref: stdapi/routes/openai_organization_usage.py
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

#: Every test drives the in-process app against a stub, so a remote target
#: would re-run them locally and report coverage it never obtained.
pytestmark = pytest.mark.local

#: One day, in seconds.
_DAY = 86400


class _FakePaginator:
    """Stand-in for the ``list_metrics`` paginator, over a fixed listing."""

    def __init__(self, metrics: list[dict[str, Any]]) -> None:
        self._metrics = metrics

    def paginate(self, **_kwargs: Any) -> Any:  # noqa: ANN401
        """Yield the single page of the fixed listing.

        Returns:
            An asynchronous iterator over one page.
        """

        async def _pages() -> Any:  # noqa: ANN401
            yield {"Metrics": self._metrics}

        return _pages()


class _FakeCloudWatch:
    """Stand-in for the CloudWatch client, recording what it was asked."""

    def __init__(
        self,
        listed: list[dict[str, Any]] | None = None,
        results: list[dict[str, Any]] | None = None,
    ) -> None:
        self.listed = listed if listed is not None else []
        self.results = results if results is not None else []
        self.queries: list[dict[str, Any]] = []
        #: One ``(results, NextToken)`` pair per call when the read paginates;
        #: the last pair answers every further call, so a pair still carrying a
        #: token is a backend that never stops paginating.
        self.pages: list[tuple[list[dict[str, Any]], str | None]] | None = None

    def get_paginator(self, _name: str) -> _FakePaginator:
        """Return the listing paginator.

        Returns:
            The stub paginator.
        """
        return _FakePaginator(self.listed)

    async def get_metric_data(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Answer with the fixed results, recording the query.

        Returns:
            The stubbed response.
        """
        self.queries.append(kwargs)
        if self.pages is None:
            return {"MetricDataResults": self.results}
        results, token = self.pages[min(len(self.queries), len(self.pages)) - 1]
        response: dict[str, Any] = {"MetricDataResults": results}
        if token:
            response["NextToken"] = token
        return response


def _series(
    metric: str, label: str, points: dict[int, float], query: str = "q0"
) -> dict[str, Any]:
    """Build one ``GetMetricData`` result.

    Args:
        metric: Unused, kept so a caller reads what the series carries.
        label: The resolved dynamic label, ``|``-separated dimension values.
        points: Value per bucket start, in Unix seconds.
        query: The query id the series answers, which names the metric.

    Returns:
        A ``MetricDataResults`` entry.
    """
    del metric
    return {
        "Id": query,
        "Label": label,
        "Timestamps": [datetime.fromtimestamp(t, UTC) for t in points],
        "Values": list(points.values()),
    }


@pytest.fixture
def cloudwatch(monkeypatch: pytest.MonkeyPatch) -> _FakeCloudWatch:
    """Install a stub CloudWatch client and an empty response cache.

    Returns:
        The stub, so a test can set its listing and results and read the
        queries the routes built.
    """
    from stdapi.aws import _CLIENTS  # noqa: PLC0415
    from stdapi.aws_cloudwatch import metrics_region  # noqa: PLC0415
    from stdapi.routes import openai_organization_usage as routes  # noqa: PLC0415

    client = _FakeCloudWatch()
    monkeypatch.setitem(_CLIENTS, "cloudwatch", {metrics_region(): client})
    monkeypatch.setattr(routes, "_CACHE", type(routes._CACHE)())  # noqa: SLF001
    return client


@pytest.fixture
def recent_start() -> int:
    """Return a start time three days ago, aligned to the daily bucket grid."""
    now = int(datetime.now(UTC).timestamp())
    return (now - 3 * _DAY) // _DAY * _DAY


def _get(
    client: TestClient,
    endpoint: str,
    **params: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Call one usage endpoint.

    Returns:
        The HTTP response.
    """
    return client.get(f"/v1/organization/usage/{endpoint}", params=params)


class _FixedVar:
    """Stand-in for a request-scoped context variable holding a fixed value."""

    def __init__(self, value: Any) -> None:  # noqa: ANN401
        self._value = value

    def get(self, *_default: Any) -> Any:  # noqa: ANN401
        """Return the fixed value.

        Returns:
            The value the test installed.
        """
        return self._value


def _as_caller(monkeypatch: pytest.MonkeyPatch, name: str, value: Any) -> None:  # noqa: ANN401
    """Make every request read a fixed principal or tenant.

    The dependency clears both on entry, and the test client runs the app in
    its own context, so the variable itself cannot be set from a test.

    Args:
        monkeypatch: The patcher.
        name: "PRINCIPAL" or "TENANT", as the route module imported it.
        value: The identity every request is to read.
    """
    from stdapi.routes import openai_organization_usage as routes  # noqa: PLC0415

    monkeypatch.setattr(routes, name, _FixedVar(value))


def _arm_deployment_key(monkeypatch: pytest.MonkeyPatch, api_key: str) -> None:
    """Hash the deployment API key the app lifespan would normally hash.

    Without it, a test that turns tenant keys on leaves the only enabled
    method the tenant one, which refuses the deployment's own key.

    Args:
        monkeypatch: The patcher.
        api_key: The key the test client presents.
    """
    from pydantic import SecretStr  # noqa: PLC0415

    import stdapi.auth  # noqa: PLC0415

    handler = stdapi.auth.AuthenticationHandler()
    handler._hash_api_key(SecretStr(api_key))  # noqa: SLF001
    monkeypatch.setattr(stdapi.auth, "_auth_handler", handler)


class TestEnvelope:
    """The page/bucket/result envelope every endpoint answers with."""

    def test_page_parses_as_the_sdk_usage_response(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """The answer is what ``client.admin.organization.usage`` expects.

        Ref: openai.types.admin.organization.usage_completions_response
        """
        from openai.types.admin.organization.usage_completions_response import (  # noqa: PLC0415
            UsageCompletionsResponse,
        )

        cloudwatch.listed = [
            {
                "MetricName": "InputTokens",
                "Dimensions": [
                    {"Name": "Model", "Value": "amazon.nova-micro-v1:0"},
                    {"Name": "Operation", "Value": "chat.completions"},
                ],
            }
        ]
        cloudwatch.results = [
            _series(
                "InputTokens",
                "amazon.nova-micro-v1:0|chat.completions",
                {recent_start: 120.0},
            )
        ]
        response = _get(app_client, "completions", start_time=recent_start, limit=3)
        assert response.status_code == 200, response.text
        page = UsageCompletionsResponse.model_validate(response.json())
        assert page.object == "page"
        assert page.data[0].object == "bucket"
        assert page.data[0].start_time == recent_start
        assert page.data[0].end_time == recent_start + _DAY

    def test_bucket_grid_is_contiguous_and_limited(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """Every bucket of a page follows the previous one, with no gap."""
        response = _get(app_client, "completions", start_time=recent_start, limit=2)
        buckets = response.json()["data"]
        assert len(buckets) == 2
        assert buckets[1]["start_time"] == buckets[0]["end_time"]

    def test_start_time_is_required(self, app_client: TestClient) -> None:
        """Upstream declares ``start_time`` required, so a query without it fails."""
        assert app_client.get("/v1/organization/usage/completions").status_code == 400


class TestMeasurements:
    """What the reported figures are built from."""

    def test_tokens_and_requests_come_from_their_own_metrics(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """``num_model_requests`` is a recorded count, never derived from tokens.

        The three input buckets are metered apart, the way the backend reports
        them, and ``input_tokens`` is declared to cover all three: an
        administrator reconciling it against the ``prompt_tokens`` the same
        request returned has to read the same number.

        Ref: stdapi/types/openai_organization_usage.py:CompletionsResult
        """
        label = "amazon.nova-micro-v1:0|chat.completions"
        cloudwatch.results = [
            _series("InputTokens", label, {recent_start: 100.0}, "q0"),
            _series("OutputTokens", label, {recent_start: 20.0}, "q1"),
            _series("CachedTokens", label, {recent_start: 30.0}, "q2"),
            _series("CacheWriteTokens", label, {recent_start: 5.0}, "q3"),
            _series("Requests", label, {recent_start: 4.0}, "q4"),
        ]
        cloudwatch.listed = [{"MetricName": "InputTokens", "Dimensions": []}]
        result = _get(
            app_client, "completions", start_time=recent_start, limit=1
        ).json()
        row = result["data"][0]["results"][0]
        assert row["input_tokens"] == 135
        assert row["output_tokens"] == 20
        assert row["input_cached_tokens"] == 30
        assert row["input_cache_write_tokens"] == 5
        assert row["input_uncached_tokens"] == 105
        assert row["num_model_requests"] == 4

    def test_grouping_by_model_reports_one_row_per_model(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """Each model's series becomes its own result, named by the dimension."""
        cloudwatch.results = [
            _series("InputTokens", "model.a|chat.completions", {recent_start: 1.0}),
            _series("InputTokens", "model.b|chat.completions", {recent_start: 2.0}),
        ]
        rows = _get(
            app_client,
            "completions",
            start_time=recent_start,
            limit=1,
            group_by="model",
        ).json()["data"][0]["results"]
        assert [row["model"] for row in rows] == ["model.a", "model.b"]

    def test_an_ungrouped_result_names_no_key(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """Upstream leaves every grouping key null unless it was asked for."""
        cloudwatch.results = [
            _series("InputTokens", "model.a|chat.completions", {recent_start: 1.0}),
            _series("InputTokens", "model.b|chat.completions", {recent_start: 2.0}),
        ]
        rows = _get(app_client, "completions", start_time=recent_start, limit=1).json()[
            "data"
        ][0]["results"]
        assert len(rows) == 1
        assert "model" not in rows[0]
        assert rows[0]["input_tokens"] == 3

    def test_a_label_that_does_not_carry_the_dimensions_is_dropped(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """A series is dropped rather than attributed to the wrong model."""
        cloudwatch.results = [
            _series("InputTokens", "unexpected label", {recent_start: 5.0})
        ]
        rows = _get(app_client, "completions", start_time=recent_start, limit=1).json()[
            "data"
        ][0]["results"]
        assert rows == []

    def test_image_source_is_read_from_the_operation(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """``group_by=source`` reports upstream's own source names."""
        cloudwatch.results = [
            _series("OutputImages", "model.a|images.generations", {recent_start: 2.0})
        ]
        rows = _get(
            app_client, "images", start_time=recent_start, limit=1, group_by="source"
        ).json()["data"][0]["results"]
        assert rows[0]["source"] == "image.generation"
        assert rows[0]["images"] == 2

    def test_web_searches_are_counted_from_the_grounding_metric(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """Only the built-in search tool's own metric counts as a web search.

        The tool is recorded as grounding requests, on the searching model for
        a native grounding tool and on a synthetic model for the Mantle one.
        ``SearchUnits`` is a different measurement -- rerank units and managed
        knowledge base retrievals -- and is not read here at all.

        Ref: stdapi/usage.py:record_web_search_usage
             stdapi/usage.py:record_knowledge_base_usage
             stdapi/models/rerank/bedrock_rerank.py
        """
        cloudwatch.results = [
            _series(
                "GroundingRequests", "search.model|responses", {recent_start: 3.0}, "q0"
            ),
            _series("Requests", "search.model|responses", {recent_start: 3.0}, "q1"),
            _series("Requests", "chat.model|responses", {recent_start: 9.0}, "q1"),
        ]
        rows = _get(
            app_client, "web_search_calls", start_time=recent_start, limit=1
        ).json()["data"][0]["results"]
        assert rows[0]["num_requests"] == 3
        assert rows[0]["num_model_requests"] == 3

    def test_rerank_and_retrieval_units_are_never_read_as_web_searches(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """The query asks for neither the metric nor the operations they carry.

        ``SearchUnits`` is what Cohere Rerank and a managed knowledge base
        retrieval publish, under the ``rerank`` and ``vector_stores.search``
        operations; a deployment that only reranks must report no web search.

        Ref: stdapi/models/rerank/bedrock_rerank.py
             stdapi/vector_stores/knowledge_base.py
        """
        _get(app_client, "web_search_calls", start_time=recent_start, limit=1)
        expressions = [
            query["Expression"] for query in cloudwatch.queries[0]["MetricDataQueries"]
        ]
        assert not any("SearchUnits" in expression for expression in expressions)
        assert all("Operation=(" in expression for expression in expressions)
        assert not any(
            operation in expression
            for expression in expressions
            for operation in ("rerank", "vector_stores.search")
        )

    def test_num_model_requests_counts_the_whole_model_when_traffic_is_mixed(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """Documented limitation: grounding shares its series with all other traffic.

        ``GroundingRequests`` is recorded on the same usage record as every
        other call to the model, so a bucket mixing searching and
        non-searching requests to one model publishes one ``Requests`` series
        the two cannot be told apart in -- ``num_model_requests`` reports the
        model's whole traffic, not the count of calls that actually searched.

        Ref: docs/api_openai_organization_usage.md#differences-from-upstream
             stdapi/routes/openai_organization_usage.py:_build_page
        """
        cloudwatch.results = [
            _series(
                "GroundingRequests", "chat.model|responses", {recent_start: 10.0}, "q0"
            ),
            _series("Requests", "chat.model|responses", {recent_start: 1000.0}, "q1"),
        ]
        rows = _get(
            app_client, "web_search_calls", start_time=recent_start, limit=1
        ).json()["data"][0]["results"]
        assert rows[0]["num_requests"] == 10
        assert rows[0]["num_model_requests"] == 1000

    def test_a_record_made_beside_the_request_is_not_a_model_request(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """A guardrail applied around an embedding is not an embedding request.

        Every usage record counts one invocation, and a guardrail policy, the
        built-in web search, a managed knowledge base retrieval and the
        translation behind an audio translation each write their own record
        under the operation of the request they served. Counting those would
        report four embedding requests for one call with a three-policy
        guardrail, and list them as token-less models of their own.

        Ref: stdapi/usage.py:record_guardrail_policy_usage
             docs/api_openai_organization_usage.md#differences-from-upstream
        """
        cloudwatch.results = [
            _series(
                "InputTokens", "embed.model|embeddings", {recent_start: 90.0}, "q0"
            ),
            _series("Requests", "embed.model|embeddings", {recent_start: 1.0}, "q1"),
            _series(
                "Requests",
                "amazon.bedrock-runtime-guardrail-topic|embeddings",
                {recent_start: 1.0},
                "q1",
            ),
            _series(
                "Requests",
                "amazon.bedrock-runtime-guardrail-content|embeddings",
                {recent_start: 1.0},
                "q1",
            ),
        ]
        rows = _get(
            app_client, "embeddings", start_time=recent_start, limit=1, group_by="model"
        ).json()["data"][0]["results"]
        assert [row["model"] for row in rows] == ["embed.model"]
        assert rows[0]["num_model_requests"] == 1
        assert rows[0]["input_tokens"] == 90

    def test_a_translated_transcription_counts_one_request_not_two(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """``/v1/audio/translations`` records Transcribe and Translate apart.

        Both land under the ``audio.translations`` operation, but only the
        transcription measures the seconds this endpoint reports.

        Ref: stdapi/usage.py:record_translate_usage
        """
        cloudwatch.results = [
            _series(
                "InputSeconds",
                "amazon.transcribe|audio.translations",
                {recent_start: 12.0},
                "q0",
            ),
            _series(
                "Requests",
                "amazon.transcribe|audio.translations",
                {recent_start: 1.0},
                "q1",
            ),
            _series(
                "Requests",
                "amazon.translate|audio.translations",
                {recent_start: 1.0},
                "q1",
            ),
        ]
        rows = _get(
            app_client, "audio_transcriptions", start_time=recent_start, limit=1
        ).json()["data"][0]["results"]
        assert rows[0]["seconds"] == 12
        assert rows[0]["num_model_requests"] == 1

    def test_every_moderation_is_counted_whatever_answered_it(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """A guardrail and Amazon Comprehend are moderating backends, not filters.

        Neither publishes input tokens, so ``moderations`` counts every record
        under its operation -- the one endpoint where the invocation count does
        not follow the measured quantity.

        Ref: stdapi/usage.py:record_guardrail_usage
             stdapi/usage.py:record_comprehend_usage
        """
        cloudwatch.results = [
            _series(
                "Requests",
                "amazon.comprehend-toxicity|moderations",
                {recent_start: 4.0},
                "q1",
            )
        ]
        rows = _get(app_client, "moderations", start_time=recent_start, limit=1).json()[
            "data"
        ][0]["results"]
        assert rows[0]["num_model_requests"] == 4


class TestUnmeasured:
    """Endpoints upstream publishes that this server measures nothing for."""

    @pytest.mark.parametrize("endpoint", ["vector_stores", "code_interpreter_sessions"])
    def test_an_unmeasured_endpoint_answers_a_well_formed_empty_page(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        endpoint: str,
    ) -> None:
        """An empty page, never a 404: the endpoint exists and answers nothing."""
        response = _get(app_client, endpoint, start_time=recent_start, limit=2)
        assert response.status_code == 200, response.text
        page = response.json()
        assert page["object"] == "page"
        assert [bucket["results"] for bucket in page["data"]] == [[], []]
        assert cloudwatch.queries == []


class TestRefusals:
    """Every parameter upstream declares that this server cannot answer."""

    @pytest.mark.parametrize(
        ("endpoint", "key"),
        [
            ("completions", "project_id"),
            ("completions", "batch"),
            ("completions", "service_tier"),
            ("images", "size"),
            ("web_search_calls", "context_level"),
            ("file_search_calls", "vector_store_id"),
        ],
    )
    def test_a_grouping_key_with_no_measurement_is_refused_with_its_reason(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        endpoint: str,
        key: str,
    ) -> None:
        """The refusal names the key and says what is not recorded."""
        response = _get(app_client, endpoint, start_time=recent_start, group_by=key)
        assert response.status_code == 400, response.text
        assert key in response.json()["error"]["message"]
        assert cloudwatch.queries == []

    @pytest.mark.parametrize(
        ("endpoint", "param", "value"),
        [
            ("completions", "project_ids", "proj_1"),
            ("completions", "batch", "true"),
            ("images", "sizes", "1024x1024"),
            ("web_search_calls", "context_levels", "low"),
            ("file_search_calls", "vector_store_ids", "vs_1"),
        ],
    )
    def test_a_filter_with_no_measurement_is_refused(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        endpoint: str,
        param: str,
        value: str,
    ) -> None:
        """A filter *is* the request, so it is refused rather than ignored."""
        response = _get(app_client, endpoint, start_time=recent_start, **{param: value})
        assert response.status_code == 400, response.text
        assert param in response.json()["error"]["message"]

    def test_grouping_by_api_key_is_refused_without_tenant_keys(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """The refusal names the setting that would make the key exist."""
        response = _get(
            app_client, "completions", start_time=recent_start, group_by="api_key_id"
        )
        assert response.status_code == 400
        assert "tenant_api_keys" in response.json()["error"]["message"]

    def test_grouping_by_user_is_refused_without_the_user_dimension(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """The refusal names the setting that would record the caller."""
        response = _get(
            app_client, "completions", start_time=recent_start, group_by="user_id"
        )
        assert response.status_code == 400
        assert (
            "cloudwatch_metrics_user_dimension" in response.json()["error"]["message"]
        )

    def test_grouping_by_both_identities_at_once_is_refused(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        api_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Refused rather than answered empty: no series is stored per pair."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "tenant_api_keys", True)
        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics_user_dimension", True)
        _arm_deployment_key(monkeypatch, api_key)
        response = app_client.get(
            "/v1/organization/usage/completions",
            params={"start_time": recent_start, "group_by": ["api_key_id", "user_id"]},
        )
        assert response.status_code == 400, response.text
        assert cloudwatch.queries == []

    @pytest.mark.parametrize(
        "params",
        [
            {"api_key_ids": ["key-a"], "user_ids": ["bob"]},
            {"api_key_ids": ["key-a"], "group_by": ["user_id"]},
            {"user_ids": ["bob"], "group_by": ["api_key_id"]},
        ],
    )
    def test_naming_both_identities_at_once_is_refused(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        api_key: str,
        params: dict[str, list[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Filtering by both is refused too, however the two are named.

        The pair is stored under no series, so an unrefused query matches
        nothing and answers a page of zeroes -- an administrator reconciling a
        key against a user would read that as usage nobody incurred.

        Ref: stdapi/routes/openai_organization_usage.py:_refuse_caller_pair
        """
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "tenant_api_keys", True)
        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics_user_dimension", True)
        _arm_deployment_key(monkeypatch, api_key)

        response = app_client.get(
            "/v1/organization/usage/completions",
            params={"start_time": recent_start, **params},
        )

        assert response.status_code == 400, response.text
        assert "never per pair" in response.json()["error"]["message"]
        assert cloudwatch.queries == []

    def test_grouping_by_api_key_is_served_when_tenant_keys_are_issued(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        api_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The tenant key of a request is a real dimension once keys are issued."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "tenant_api_keys", True)
        _arm_deployment_key(monkeypatch, api_key)
        cloudwatch.results = [
            _series("InputTokens", "model.a|chat.completions|key1", {recent_start: 7.0})
        ]
        rows = _get(
            app_client,
            "completions",
            start_time=recent_start,
            limit=1,
            group_by="api_key_id",
        ).json()["data"][0]["results"]
        assert rows[0]["api_key_id"] == "key1"
        assert "ApiKey" in cloudwatch.queries[0]["MetricDataQueries"][0]["Expression"]

    def test_grouping_by_user_is_served_when_the_dimension_is_published(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Turning the user dimension on is what makes ``user_id`` answerable."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics_user_dimension", True)
        cloudwatch.results = [
            _series("InputTokens", "model.a|chat.completions|bob", {recent_start: 7.0})
        ]
        rows = _get(
            app_client,
            "completions",
            start_time=recent_start,
            limit=1,
            group_by="user_id",
        ).json()["data"][0]["results"]
        assert rows[0]["user_id"] == "bob"


class TestQueryBounds:
    """What stops a query from costing more than the operator agreed to."""

    def test_a_narrow_bucket_over_an_old_range_is_refused_with_its_limit(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch
    ) -> None:
        """Refused rather than answered at a resolution nobody asked for."""
        now = int(datetime.now(UTC).timestamp())
        response = _get(
            app_client,
            "completions",
            start_time=now - 20 * _DAY,
            end_time=now - 19 * _DAY,
            bucket_width="1m",
        )
        assert response.status_code == 400, response.text
        assert "15 days" in response.json()["error"]["message"]
        assert cloudwatch.queries == []

    def test_the_same_range_is_served_at_a_wider_bucket(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch
    ) -> None:
        """The refusal is about the requested resolution, not about the range."""
        now = int(datetime.now(UTC).timestamp())
        response = _get(
            app_client,
            "completions",
            start_time=now - 20 * _DAY,
            end_time=now - 19 * _DAY,
            bucket_width="1d",
        )
        assert response.status_code == 200, response.text

    def test_a_range_longer_than_the_maximum_is_refused(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch
    ) -> None:
        """The bound on the span is what bounds the response and the charge."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        now = int(datetime.now(UTC).timestamp())
        response = _get(
            app_client,
            "completions",
            start_time=now - (SETTINGS.usage_api_max_range_days + 5) * _DAY,
        )
        assert response.status_code == 400, response.text
        assert (
            str(SETTINGS.usage_api_max_range_days)
            in (response.json()["error"]["message"])
        )

    def test_a_query_matching_too_many_series_is_refused_before_it_is_billed(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The listing is free; the read it guards is not."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "usage_api_max_metrics", 1)
        cloudwatch.listed = [
            {
                "MetricName": "InputTokens",
                "Dimensions": [
                    {"Name": "Model", "Value": f"model.{index}"},
                    {"Name": "Operation", "Value": "chat.completions"},
                ],
            }
            for index in range(2)
        ]
        response = _get(app_client, "completions", start_time=recent_start)
        assert response.status_code == 400, response.text
        assert cloudwatch.queries == []

    def test_the_series_counted_are_the_ones_the_query_asks_for(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The refusal tells the caller to narrow the query, so narrowing works.

        The count is taken under the query's own filters and operations: a
        crowded deployment refuses the wide query and admits the same query
        restricted to one model, which reads the two series it names. Counting
        the whole namespace refused both, with a remedy that could not work.

        Ref: docs/operations_cost_management.md#usage-api-cost
        """
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "usage_api_max_metrics", 4)
        cloudwatch.listed = [
            {
                "MetricName": metric,
                "Dimensions": [
                    {"Name": "Model", "Value": f"model.{index}"},
                    {"Name": "Operation", "Value": "chat.completions"},
                ],
            }
            for index in range(3)
            for metric in ("InputTokens", "Requests")
        ]
        assert _get(app_client, "completions", start_time=recent_start).status_code == (
            400
        )
        assert cloudwatch.queries == []
        narrowed = _get(
            app_client, "completions", start_time=recent_start, models="model.1"
        )
        assert narrowed.status_code == 200, narrowed.text
        assert len(cloudwatch.queries) == 1

    def test_another_endpoints_series_do_not_count_against_this_one(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``Requests`` and ``InputTokens`` are published under every operation.

        An embeddings query reads only the embeddings operation, so the chat
        traffic listed under the same metric names is not charged to it.
        """
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "usage_api_max_metrics", 1)
        cloudwatch.listed = [
            {
                "MetricName": "InputTokens",
                "Dimensions": [
                    {"Name": "Model", "Value": f"model.{index}"},
                    {"Name": "Operation", "Value": "chat.completions"},
                ],
            }
            for index in range(5)
        ]
        response = _get(app_client, "embeddings", start_time=recent_start, limit=1)
        assert response.status_code == 200, response.text

    def test_a_repeated_query_is_answered_from_the_cache(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """A client polling faster than the bucket width is not billed twice."""
        _get(app_client, "completions", start_time=recent_start, limit=1)
        _get(app_client, "completions", start_time=recent_start, limit=1)
        assert len(cloudwatch.queries) == 1

    def test_the_cache_is_not_used_when_the_ttl_is_zero(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The operator can turn the cache off."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "usage_api_cache_ttl", 0)
        _get(app_client, "completions", start_time=recent_start, limit=1)
        _get(app_client, "completions", start_time=recent_start, limit=1)
        assert len(cloudwatch.queries) == 2

    def test_only_the_requested_page_is_read(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """A page reads its own buckets, never the whole requested range."""
        _get(app_client, "completions", start_time=recent_start, limit=1)
        query = cloudwatch.queries[0]
        assert query["EndTime"] - query["StartTime"] == _DAY

    def test_a_model_filter_that_is_not_a_model_is_refused(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """A caller value never reaches the query language unchecked."""
        response = _get(
            app_client,
            "completions",
            start_time=recent_start,
            models='" OR MetricName="Cost',
        )
        assert response.status_code == 400, response.text
        assert cloudwatch.queries == []

    @pytest.mark.parametrize(
        ("width", "period"), [("1m", 60), ("1h", 3600), ("1d", _DAY)]
    )
    def test_the_bucket_width_is_the_period_the_metrics_are_read_at(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        width: str,
        period: int,
    ) -> None:
        """A bucket is one aggregation period, not a client-side re-bucketing."""
        now = int(datetime.now(UTC).timestamp())
        _get(
            app_client,
            "completions",
            start_time=now - 2 * period,
            bucket_width=width,
            limit=1,
        )
        assert cloudwatch.queries[0]["MetricDataQueries"][0]["Period"] == period

    @pytest.mark.parametrize(("width", "maximum"), [("1h", 168), ("1d", 31)])
    def test_a_page_wider_than_upstream_allows_is_refused(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        width: str,
        maximum: int,
    ) -> None:
        """Upstream bounds the page size per bucket width; so does this.

        The message has to name the per-width maximum: the schema's own
        ``le=1440`` refuses anything above the widest of the three whatever the
        width, so a bare 400 would not tell the two refusals apart. ``1m`` is
        the width the schema alone covers, and is asserted separately.

        Ref: stdapi/routes/openai_organization_usage.py:_resolve_limit
        """
        response = _get(
            app_client,
            "completions",
            start_time=recent_start,
            bucket_width=width,
            limit=maximum + 1,
        )
        assert response.status_code == 400, response.text
        assert (
            f"'limit' may not exceed {maximum}" in response.json()["error"]["message"]
        )

    def test_a_page_wider_than_every_width_allows_is_refused(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """The widest maximum bounds the parameter itself, before any handler runs."""
        response = _get(
            app_client,
            "completions",
            start_time=recent_start,
            bucket_width="1m",
            limit=1441,
        )
        assert response.status_code == 400, response.text
        assert cloudwatch.queries == []


class TestBackendPagination:
    """Reading a range CloudWatch answers over more than one page."""

    def test_every_page_of_a_read_is_merged_into_the_report(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """A continued read reports both pages, and carries the token back.

        CloudWatch caps one ``GetMetricData`` response at 100 800 data points
        and hands back a ``NextToken`` for the rest. Dropping the later pages
        would answer a partial total with a 200 -- a wrong number, not an error.

        Ref: https://docs.aws.amazon.com/AmazonCloudWatch/latest/APIReference/API_GetMetricData.html
        """
        cloudwatch.pages = [
            (
                [
                    _series(
                        "InputTokens", "model.a|chat.completions", {recent_start: 1.0}
                    )
                ],
                "page-2",
            ),
            (
                [
                    _series(
                        "InputTokens", "model.b|chat.completions", {recent_start: 2.0}
                    )
                ],
                None,
            ),
        ]
        rows = _get(
            app_client,
            "completions",
            start_time=recent_start,
            limit=1,
            group_by="model",
        ).json()["data"][0]["results"]

        assert [row["model"] for row in rows] == ["model.a", "model.b"]
        assert len(cloudwatch.queries) == 2
        assert cloudwatch.queries[1]["NextToken"] == "page-2"

    def test_a_read_that_never_stops_paginating_is_refused(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """Past the page cap the query is refused, never truncated silently.

        The alternative is answering the pages already read as if they were the
        whole range: a usage figure that is wrong, answered 200, with nothing
        for the caller to notice it by.

        Ref: stdapi/aws_cloudwatch.py:read_series
        """
        from stdapi.aws_cloudwatch import _MAX_PAGES  # noqa: PLC0415

        cloudwatch.pages = [
            (
                [
                    _series(
                        "InputTokens", "model.a|chat.completions", {recent_start: 1.0}
                    )
                ],
                "more",
            )
        ]

        response = _get(app_client, "completions", start_time=recent_start, limit=1)

        assert response.status_code == 400, response.text
        assert "Narrow it" in response.json()["error"]["message"]
        assert len(cloudwatch.queries) == _MAX_PAGES


class TestPagination:
    """The cursor a client follows to read a long range."""

    def test_the_next_page_continues_where_the_first_stopped(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch
    ) -> None:
        """Following the cursor walks the bucket grid without repeating one."""
        now = int(datetime.now(UTC).timestamp())
        start = (now - 5 * _DAY) // _DAY * _DAY
        first = _get(app_client, "completions", start_time=start, limit=2).json()
        assert first["has_more"] is True
        second = _get(
            app_client,
            "completions",
            start_time=start,
            limit=2,
            page=first["next_page"],
        ).json()
        assert second["data"][0]["start_time"] == first["data"][-1]["end_time"]

    def test_the_cursor_carries_no_backend_token(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch
    ) -> None:
        """The cursor is this server's own, so it never leaks a backend one."""
        from base64 import urlsafe_b64decode  # noqa: PLC0415

        now = int(datetime.now(UTC).timestamp())
        start = (now - 5 * _DAY) // _DAY * _DAY
        cursor = _get(app_client, "completions", start_time=start, limit=2).json()[
            "next_page"
        ]
        assert urlsafe_b64decode(cursor.encode()).decode() == "2"

    def test_a_cursor_this_server_did_not_issue_is_refused(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """A backend token pasted in as a cursor is refused, never followed."""
        response = _get(
            app_client, "completions", start_time=recent_start, page="not-a-cursor"
        )
        assert response.status_code == 400, response.text


class TestCosts:
    """The costs endpoint, and the two settings it needs."""

    def test_costs_need_cost_tracking(
        self, app_client: TestClient, cloudwatch: _FakeCloudWatch, recent_start: int
    ) -> None:
        """Without cost tracking there is no cost recorded to report."""
        response = app_client.get(
            "/v1/organization/costs", params={"start_time": recent_start}
        )
        assert response.status_code == 503, response.text

    def test_costs_are_reported_per_currency(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Currencies are never summed together into one meaningless number."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        cloudwatch.results = [
            _series("Cost", "model.a|USD", {recent_start: 1.5}),
            _series("Cost", "model.b|EUR", {recent_start: 2.5}),
        ]
        response = app_client.get(
            "/v1/organization/costs", params={"start_time": recent_start, "limit": 1}
        )
        assert response.status_code == 200, response.text
        rows = response.json()["data"][0]["results"]
        assert {row["amount"]["currency"] for row in rows} == {"usd", "eur"}
        assert all(row["object"] == "organization.costs.result" for row in rows)

    def test_costs_group_by_api_key_when_tenant_keys_are_issued(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        api_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Spend per tenant key is what a reseller needs, and the key is recorded."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        monkeypatch.setattr(SETTINGS, "tenant_api_keys", True)
        _arm_deployment_key(monkeypatch, api_key)
        cloudwatch.results = [_series("Cost", "model.a|USD|key1", {recent_start: 3.0})]
        response = app_client.get(
            "/v1/organization/costs",
            params={"start_time": recent_start, "limit": 1, "group_by": "api_key_id"},
        )
        assert response.status_code == 200, response.text
        row = response.json()["data"][0]["results"][0]
        assert row["api_key_id"] == "key1"
        assert row["amount"]["value"] == 3.0

    def test_costs_refuse_user_grouping(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Upstream does not group costs by user, and no cost is recorded per user."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics_user_dimension", True)
        response = app_client.get(
            "/v1/organization/costs",
            params={"start_time": recent_start, "group_by": "user_id"},
        )
        assert response.status_code == 400, response.text

    def test_costs_refuse_line_item_grouping(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No cost is recorded per product line, so grouping by one is refused."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        response = app_client.get(
            "/v1/organization/costs",
            params={"start_time": recent_start, "group_by": "line_item"},
        )
        assert response.status_code == 400, response.text
        assert "line_item" in response.json()["error"]["message"]


class TestAuthorization:
    """Who may read what the whole deployment consumed."""

    def test_a_tenant_key_may_not_read_the_organization(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A per-customer credential is never an administrator credential.

        Ref: stdapi/monitoring.py:TENANT
        """
        from stdapi.monitoring import Tenant  # noqa: PLC0415

        _as_caller(monkeypatch, "TENANT", Tenant(key_id="key1", name="tenant"))
        response = _get(app_client, "completions", start_time=recent_start)
        assert response.status_code == 403, response.text
        assert cloudwatch.queries == []

    def test_a_user_token_is_refused_when_no_admin_scope_is_configured(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fails closed: an end user must not read organization-wide usage."""
        from stdapi.monitoring import Principal  # noqa: PLC0415

        _as_caller(monkeypatch, "PRINCIPAL", Principal(subject="bob"))
        response = _get(app_client, "completions", start_time=recent_start)
        assert response.status_code == 403, response.text

    def test_a_bad_parameter_is_answered_after_authorization_not_before(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A caller who may not read anything learns nothing from a bad parameter.

        The refusals name the settings this deployment runs with, so answering
        400 before the 403 would let a tenant map the server's configuration by
        sending parameters it is not allowed to use.

        Ref: stdapi/routes/openai_organization_usage.py:_require_admin
        """
        from stdapi.monitoring import Tenant  # noqa: PLC0415

        _as_caller(monkeypatch, "TENANT", Tenant(key_id="key1", name="tenant"))

        response = _get(
            app_client, "completions", start_time=recent_start, project_ids="proj_1"
        )

        assert response.status_code == 403, response.text
        assert "project_ids" not in response.json()["error"]["message"]

    def test_a_user_token_carrying_every_admin_scope_is_accepted(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The scopes the operator names are what makes a token administrative."""
        from stdapi.config import SETTINGS  # noqa: PLC0415
        from stdapi.monitoring import Principal  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "usage_api_admin_scopes", ["stdapi/usage.read"])
        _as_caller(
            monkeypatch,
            "PRINCIPAL",
            Principal(subject="bob", scopes=frozenset({"stdapi/usage.read"})),
        )
        response = _get(app_client, "completions", start_time=recent_start)
        assert response.status_code == 200, response.text

    def test_a_user_token_missing_an_admin_scope_is_refused(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every named scope must be carried, not just one of them."""
        from stdapi.config import SETTINGS  # noqa: PLC0415
        from stdapi.monitoring import Principal  # noqa: PLC0415

        monkeypatch.setattr(
            SETTINGS, "usage_api_admin_scopes", ["stdapi/usage.read", "stdapi/admin"]
        )
        _as_caller(
            monkeypatch,
            "PRINCIPAL",
            Principal(subject="bob", scopes=frozenset({"stdapi/usage.read"})),
        )
        response = _get(app_client, "completions", start_time=recent_start)
        assert response.status_code == 403, response.text

    def test_a_bad_parameter_is_answered_after_authorization_on_every_endpoint(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No endpoint of the surface refuses a parameter before it refuses the caller.

        The check is per route body rather than inside the shared helper, so a
        twelfth endpoint added without it would leak the same oracle on its own.

        Ref: stdapi/routes/openai_organization_usage.py:_require_admin
        """
        from stdapi.monitoring import Tenant  # noqa: PLC0415

        _as_caller(monkeypatch, "TENANT", Tenant(key_id="key1", name="tenant"))

        for endpoint in (
            "completions",
            "embeddings",
            "moderations",
            "images",
            "audio_speeches",
            "audio_transcriptions",
            "web_search_calls",
            "file_search_calls",
            "vector_stores",
            "code_interpreter_sessions",
        ):
            response = _get(
                app_client,
                endpoint,
                start_time=recent_start,
                project_ids="proj_1",
                group_by="project_id",
            )
            assert response.status_code == 403, f"{endpoint}: {response.text}"
        costs = app_client.get(
            "/v1/organization/costs",
            params={"start_time": recent_start, "group_by": "line_item"},
        )
        assert costs.status_code == 403, costs.text
        assert cloudwatch.queries == []

    def test_the_surface_refuses_everything_when_it_is_not_enabled(
        self,
        app_client: TestClient,
        cloudwatch: _FakeCloudWatch,
        recent_start: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Off by default: the endpoints exist, and answer that they are not served.

        Ref: stdapi/api_errors.py:FeatureUnavailableError
        """
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "usage_api", False)
        response = _get(app_client, "completions", start_time=recent_start)
        assert response.status_code == 503, response.text
        assert cloudwatch.queries == []


class TestStartupWarnings:
    """What the operator is told at startup about what the surface will answer."""

    @staticmethod
    def _warnings(
        monkeypatch: pytest.MonkeyPatch,
        *,
        cloudwatch_metrics: bool = True,
        cost_tracking: bool = True,
        user_pool: str = "",
        admin_scopes: tuple[str, ...] = ("stdapi/usage.read",),
    ) -> list[str]:
        """Run the startup warnings against one configuration.

        Returns:
            The warning texts the run added to the startup event.
        """
        from stdapi.aws_cloudwatch import add_usage_api_warnings  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics", cloudwatch_metrics)
        monkeypatch.setattr(SETTINGS, "cost_tracking", cost_tracking)
        monkeypatch.setattr(SETTINGS, "aws_cognito_user_pool_id", user_pool)
        monkeypatch.setattr(SETTINGS, "usage_api_admin_scopes", list(admin_scopes))
        event: Any = {"level": "info"}
        add_usage_api_warnings(event)
        return [str(warning) for warning in event.get("server_warnings", [])]

    def test_metrics_off_is_reported_as_nothing_to_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The usage endpoints read what ``cloudwatch_metrics`` publishes.

        Cost tracking is off here too, and is deliberately *not* warned about a
        second time: without the metrics there is nothing for a cost to be
        attached to, so naming both would make the operator fix the wrong one.
        """
        warnings = self._warnings(
            monkeypatch, cloudwatch_metrics=False, cost_tracking=False
        )
        assert len(warnings) == 1
        assert "'cloudwatch_metrics'" in warnings[0]

    def test_cost_tracking_off_is_reported_for_the_costs_endpoint_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the costs endpoint needs it, so only it is named."""
        warnings = self._warnings(monkeypatch, cost_tracking=False)
        assert len(warnings) == 1
        assert "'cost_tracking'" in warnings[0]
        assert "costs endpoint" in warnings[0]

    def test_a_user_pool_without_admin_scopes_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise the operator meets the closed door as a 403 in production.

        Ref: stdapi/routes/openai_organization_usage.py:_require_admin
        """
        warnings = self._warnings(
            monkeypatch, user_pool="eu-west-3_abc123", admin_scopes=()
        )
        assert len(warnings) == 1
        assert "'usage_api_admin_scopes'" in warnings[0]

    def test_a_fully_configured_deployment_is_warned_about_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A warning the operator cannot act on trains them to ignore the rest."""
        assert self._warnings(monkeypatch, user_pool="eu-west-3_abc123") == []
