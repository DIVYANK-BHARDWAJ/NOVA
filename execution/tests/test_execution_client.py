"""Unit tests for TradingApiClient with mocked HTTP responses."""

import io
import json
import socket
import urllib.error
from unittest.mock import MagicMock, patch
import pytest

from execution.client import (
    AuthenticationError,
    AuthorizationError,
    InvalidJsonResponseError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    TradingApiClient,
    TradingApiError,
)
from execution.config import ExecutionConfig


@pytest.fixture
def test_config() -> ExecutionConfig:
    return ExecutionConfig(
        bot_access_token="test-secret-token-abcdef",
        api_base_url="http://192.168.65.157:3002",
        http_timeout_seconds=3.0,
    )


@pytest.fixture
def client(test_config: ExecutionConfig) -> TradingApiClient:
    return TradingApiClient(test_config)


def _make_mock_response(status: int, body_data: dict | list | str) -> MagicMock:
    """Create a mock HTTP response context manager."""
    if isinstance(body_data, (dict, list)):
        payload_bytes = json.dumps(body_data).encode("utf-8")
    else:
        payload_bytes = body_data.encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.code = status
    mock_resp.read.return_value = payload_bytes
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


def test_get_stage_success(client: TradingApiClient):
    """get_stage correctly parses response into SimulationStage model."""
    mock_data = {
        "stage": "development",
        "status": "running",
        "tickers": ["AERX", "ROUT"],
        "ticker_count": 2,
        "current_date": "2026-03-25",
        "days_completed": 353,
        "total_days": 500,
        "interval_seconds": 10,
        "is_sandbox_stream": True,
        "sandbox_loop": 1,
    }

    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, mock_data)):
        stage = client.get_stage()
        assert stage.stage == "development"
        assert stage.status == "running"
        assert stage.is_trading_active is True
        assert stage.tickers == ["AERX", "ROUT"]
        assert stage.days_completed == 353
        assert stage.total_days == 500


def test_get_stocks_success(client: TradingApiClient):
    """get_stocks parses stock array into StockInfo models."""
    mock_data = {
        "status": "ok",
        "stocks": [
            {"symbol": "ROUT", "ticker": "ROUT", "name": "RouteWave Mobility", "description": "Mobility"},
            {"symbol": "BLUX", "ticker": "BLUX", "name": "BlueMatrix", "description": "Cloud"},
        ],
    }

    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, mock_data)):
        stocks = client.get_stocks()
        assert len(stocks) == 2
        assert stocks[0].ticker == "ROUT"
        assert stocks[0].name == "RouteWave Mobility"
        assert stocks[1].ticker == "BLUX"


def test_get_simulation_clock(client: TradingApiClient):
    """get_simulation_clock parses clock response."""
    mock_data = {
        "status": "running",
        "simulated_date": "2026-03-30",
        "current_day": 356,
        "days_completed": 356,
        "total_days": 500,
        "remaining_days": 144,
        "interval_seconds": 10,
    }

    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, mock_data)):
        clock = client.get_simulation_clock()
        assert clock.status == "running"
        assert clock.simulated_date == "2026-03-30"
        assert clock.days_completed == 356
        assert clock.remaining_days == 144


def test_get_portfolio_cash(client: TradingApiClient):
    """get_cash parses available cash and margin balance."""
    mock_data = {
        "team_id": "team_123",
        "team_name": "Nova Forge",
        "available_cash": 850000.5,
        "cash_balance": 900000.0,
        "capital_remaining": 850000.5,
        "margin_locked": 50000.0,
        "initial_capital": 1000000.0,
        "realized_pnl": 12500.0,
        "currency": "INR",
        "is_blocked": False,
    }

    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, mock_data)):
        cash = client.get_cash()
        assert cash.team_id == "team_123"
        assert cash.available_cash == 850000.5
        assert cash.margin_locked == 50000.0
        assert cash.currency == "INR"
        assert cash.is_blocked is False


def test_get_portfolio_with_holdings(client: TradingApiClient):
    """get_portfolio parses overall equity, P&L, and holding positions."""
    mock_data = {
        "team_id": "team_123",
        "team_name": "Nova Forge",
        "available_cash": 800000.0,
        "cash_balance": 800000.0,
        "portfolio_market_value": 210000.0,
        "total_wealth": 1010000.0,
        "realized_pnl": 5000.0,
        "unrealized_pnl": 5000.0,
        "total_pnl": 10000.0,
        "total_charges": 250.0,
        "total_taxes": 125.0,
        "total_orders": 4,
        "rejected_orders_count": 0,
        "error_count": 0,
        "is_blocked": False,
        "holdings": [
            {
                "ticker": "ROUT",
                "quantity": 100,
                "avg_price": 75.0,
                "current_price": 78.5,
                "market_value": 7850.0,
                "unrealized_pnl": 350.0,
            }
        ],
    }

    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, mock_data)):
        port = client.get_portfolio()
        assert port.available_cash == 800000.0
        assert port.total_wealth == 1010000.0
        assert len(port.holdings) == 1
        assert port.holdings[0].ticker == "ROUT"
        assert port.holdings[0].quantity == 100
        assert port.holdings[0].unrealized_pnl == 350.0


def test_get_orders_pagination_and_query_params(client: TradingApiClient):
    """get_orders correctly serializes query parameters."""
    mock_data = {
        "orders": [
            {"id": "ord_1", "ticker": "ROUT", "action": "BUY", "quantity": 50, "status": "FILLED"}
        ],
        "total": 1,
        "page": 2,
        "limit": 20,
        "totalPages": 3,
    }

    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, mock_data)) as mock_urlopen:
        res = client.get_orders(page=2, limit=20, status="FILLED", action="BUY", ticker="ROUT")
        assert res["page"] == 2
        assert len(res["orders"]) == 1

        # Check call arguments
        req = mock_urlopen.call_args[0][0]
        url = req.full_url
        assert "page=2" in url
        assert "limit=20" in url
        assert "status=FILLED" in url
        assert "action=BUY" in url
        assert "ticker=ROUT" in url


def test_submit_order_payload(client: TradingApiClient):
    """submit_order sends POST with documented payload fields."""
    mock_data = {
        "status": "FILLED",
        "order": {
            "order_id": "ord_9988",
            "ticker": "ROUT",
            "action": "BUY",
            "quantity": 50,
            "price": 78.5,
            "status": "FILLED",
            "charges": 15.0,
        },
    }

    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, mock_data)) as mock_urlopen:
        res = client.submit_order(ticker="ROUT", action="BUY", quantity=50, date="2026-03-25")
        assert res["status"] == "FILLED"

        req = mock_urlopen.call_args[0][0]
        assert req.get_method() == "POST"
        body = json.loads(req.data.decode("utf-8"))
        assert body == {"ticker": "ROUT", "action": "BUY", "quantity": 50, "date": "2026-03-25"}
        assert req.get_header("Content-type") == "application/json"
        assert req.get_header("Authorization") == "Bearer test-secret-token-abcdef"


def test_send_heartbeat(client: TradingApiClient):
    """send_heartbeat posts seq payload to /api/v1/bot/heartbeat."""
    mock_data = {"status": "ok", "acknowledged_seq": "1001"}

    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, mock_data)) as mock_urlopen:
        res = client.send_heartbeat(1001)
        assert res["status"] == "ok"

        req = mock_urlopen.call_args[0][0]
        assert req.get_method() == "POST"
        assert req.full_url.endswith("/api/v1/bot/heartbeat")
        body = json.loads(req.data.decode("utf-8"))
        assert body == {"seq": "1001"}


# ============================================================================
# HTTP Error Mapping & Exception Tests
# ============================================================================

def test_http_401_unauthorized(client: TradingApiClient):
    """HTTP 401 raises AuthenticationError without exposing token."""
    err = urllib.error.HTTPError(
        url="http://192.168.65.157:3002/api/v1/portfolio",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=io.BytesIO(b'{"error":"UNAUTHORIZED","message":"Bearer access token required"}'),
    )

    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(AuthenticationError, match="HTTP 401 Unauthorized") as exc_info:
            client.get_portfolio()

        # Token must NOT be present in exception string
        assert "test-secret-token-abcdef" not in str(exc_info.value)
        assert exc_info.value.status_code == 401


def test_http_403_account_verification_failure(client: TradingApiClient):
    """HTTP 403 raises AuthorizationError."""
    err = urllib.error.HTTPError(
        url="http://192.168.65.157:3002/api/v1/orders",
        code=403,
        msg="Forbidden",
        hdrs={},
        fp=io.BytesIO(b'{"error":"FORBIDDEN","message":"Account is blocked or unverified"}'),
    )

    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(AuthorizationError, match="HTTP 403 Forbidden") as exc_info:
            client.submit_order("ROUT", "BUY", 10)
        assert exc_info.value.status_code == 403


def test_other_http_errors_404_429_500_400(client: TradingApiClient):
    """Test 404, 429, 500, and 400 error mappings."""
    # 404
    err404 = urllib.error.HTTPError("http://...", 404, "Not Found", {}, io.BytesIO(b'{"error":"NOT_FOUND"}'))
    with patch("urllib.request.urlopen", side_effect=err404):
        with pytest.raises(NotFoundError):
            client.get_stage()

    # 429
    err429 = urllib.error.HTTPError("http://...", 429, "Too Many Requests", {}, io.BytesIO(b'{"error":"RATE_LIMIT"}'))
    with patch("urllib.request.urlopen", side_effect=err429):
        with pytest.raises(RateLimitError):
            client.get_stage()

    # 500
    err500 = urllib.error.HTTPError("http://...", 500, "Server Error", {}, io.BytesIO(b'{"error":"INTERNAL"}'))
    with patch("urllib.request.urlopen", side_effect=err500):
        with pytest.raises(ServerError):
            client.get_stage()

    # 400
    err400 = urllib.error.HTTPError("http://...", 400, "Bad Request", {}, io.BytesIO(b'{"error":"BAD_REQUEST"}'))
    with patch("urllib.request.urlopen", side_effect=err400):
        with pytest.raises(TradingApiError) as exc_info:
            client.get_stage()
        assert exc_info.value.status_code == 400


def test_connection_failure(client: TradingApiClient):
    """Socket error or URLError raises NetworkError."""
    url_err = urllib.error.URLError(reason=ConnectionRefusedError("Connection refused"))
    with patch("urllib.request.urlopen", side_effect=url_err):
        with pytest.raises(NetworkError, match="Network error"):
            client.get_stage()


def test_request_timeout(client: TradingApiClient):
    """Timeout error raises RequestTimeoutError with UNKNOWN outcome warning."""
    sock_timeout = socket.timeout("timed out")
    with patch("urllib.request.urlopen", side_effect=sock_timeout):
        with pytest.raises(RequestTimeoutError, match="UNKNOWN"):
            client.submit_order("ROUT", "BUY", 10)


def test_malformed_json_response(client: TradingApiClient):
    """Server returning non-JSON or HTML raises InvalidJsonResponseError."""
    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, "<html>Bad Gateway</html>")):
        with pytest.raises(InvalidJsonResponseError, match="invalid JSON"):
            client.get_stage()


def test_get_stock_history(client: TradingApiClient):
    """get_stock_history sends GET with ticker and limit query params."""
    mock_data = {"status": "ok", "ticker": "ROUT", "prices": []}
    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, mock_data)) as mock_urlopen:
        res = client.get_stock_history("ROUT", limit=50)
        assert res["ticker"] == "ROUT"
        req = mock_urlopen.call_args[0][0]
        assert "ticker=ROUT" in req.full_url
        assert "limit=50" in req.full_url


def test_sanitize_headers_for_logging(client: TradingApiClient):
    """_sanitize_headers_for_logging masks Authorization header."""
    headers = {"Authorization": "Bearer real-secret-token", "Accept": "application/json"}
    sanitized = client._sanitize_headers_for_logging(headers)
    assert sanitized["Authorization"] == "Bearer ***REDACTED***"
    assert sanitized["Accept"] == "application/json"


def test_invalid_json_structures(client: TradingApiClient):
    """Client raises InvalidJsonResponseError when expected dicts/lists are wrong types."""
    # /api/v1/stage returning list instead of dict
    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, [1, 2, 3])):
        with pytest.raises(InvalidJsonResponseError, match="Expected dict from /api/v1/stage"):
            client.get_stage()

    # /api/v1/stocks returning dict with 'stocks' as string
    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, {"stocks": "not-a-list"})):
        with pytest.raises(InvalidJsonResponseError, match="Expected 'stocks' list"):
            client.get_stocks()

    # /api/v1/stocks returning list instead of dict
    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, ["not-a-dict"])):
        with pytest.raises(InvalidJsonResponseError, match="Expected dict from /api/v1/stocks"):
            client.get_stocks()

    # /api/v1/simulation/clock returning list
    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, [1])):
        with pytest.raises(InvalidJsonResponseError, match="Expected dict from /api/v1/simulation/clock"):
            client.get_simulation_clock()

    # /api/v1/portfolio returning list
    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, [1])):
        with pytest.raises(InvalidJsonResponseError, match="Expected dict from /api/v1/portfolio"):
            client.get_portfolio()

    # /api/v1/portfolio/cash returning list
    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, [1])):
        with pytest.raises(InvalidJsonResponseError, match="Expected dict from /api/v1/portfolio/cash"):
            client.get_cash()


def test_portfolio_holdings_average_price_alias(client: TradingApiClient):
    """Portfolio holding parses average_price alias if avg_price is missing."""
    mock_data = {
        "available_cash": 1000.0,
        "holdings": [{"ticker": "ROUT", "quantity": 10, "average_price": 55.5}],
    }
    with patch("urllib.request.urlopen", return_value=_make_mock_response(200, mock_data)):
        port = client.get_portfolio()
        assert port.holdings[0].avg_price == 55.5
