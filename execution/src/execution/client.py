"""HTTP REST Client for MANTHAN 2026 Trading Simulation Platform.

Handles authenticated communication with the competition REST API, converts responses
into strongly-typed models, manages timeouts and error handling, and strictly protects
sensitive access tokens.
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from execution.config import ExecutionConfig
from execution.models import (
    CashBalance,
    Portfolio,
    PortfolioHolding,
    SimulationClock,
    SimulationStage,
    StockInfo,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class TradingApiError(Exception):
    """Base exception for all trading API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(TradingApiError):
    """Raised on HTTP 401 Unauthorized (invalid or missing BOT_ACCESS_TOKEN)."""
    pass


class AuthorizationError(TradingApiError):
    """Raised on HTTP 403 Forbidden (account verification or permission failure)."""
    pass


class NotFoundError(TradingApiError):
    """Raised on HTTP 404 Not Found."""
    pass


class RateLimitError(TradingApiError):
    """Raised on HTTP 429 Too Many Requests."""
    pass


class ServerError(TradingApiError):
    """Raised on HTTP 5xx Internal Server Error."""
    pass


class NetworkError(TradingApiError):
    """Raised on socket connection failures, DNS errors, or broken pipes."""
    pass


class RequestTimeoutError(TradingApiError):
    """Raised when an HTTP request times out. Indicates an UNKNOWN outcome for state changes."""
    pass


class InvalidJsonResponseError(TradingApiError):
    """Raised when the server returns a non-JSON or malformed payload."""
    pass


# ============================================================================
# Client Implementation
# ============================================================================

class TradingApiClient:
    """Robust, authenticated HTTP client for MANTHAN 2026 trading simulation."""

    def __init__(self, config: ExecutionConfig) -> None:
        """Initialize client with validated ExecutionConfig.

        Args:
            config: ExecutionConfig instance.
        """
        self.config = config
        self.config.validate()
        self._base_url = config.api_base_url.rstrip("/")
        self._timeout = config.http_timeout_seconds

    def _build_url(self, path: str, query_params: Optional[dict[str, Any]] = None) -> str:
        """Construct normalized full URL with optional query parameters."""
        clean_path = path.lstrip("/")
        full_url = f"{self._base_url}/{clean_path}"
        if query_params:
            filtered_params = {k: v for k, v in query_params.items() if v is not None}
            if filtered_params:
                full_url = f"{full_url}?{urllib.parse.urlencode(filtered_params)}"
        return full_url

    def _sanitize_headers_for_logging(self, headers: dict[str, str]) -> dict[str, str]:
        """Return headers with Authorization token redacted."""
        sanitized = dict(headers)
        if "Authorization" in sanitized:
            sanitized["Authorization"] = "Bearer ***REDACTED***"
        return sanitized

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
        query_params: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Execute HTTP request with strict error handling and token safety.

        Args:
            method: HTTP method ('GET', 'POST').
            path: Relative API path (e.g. '/api/v1/orders').
            payload: Optional JSON body.
            query_params: Optional query string parameters.

        Returns:
            Decoded JSON response object.

        Raises:
            TradingApiError: Subclass matching specific failure mode.
        """
        url = self._build_url(path, query_params)

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.config.bot_access_token}",
            "Accept": "application/json",
            "User-Agent": "NOVA-Trading-Bot/1.0",
        }

        data_bytes: Optional[bytes] = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)

        logger.debug(
            "Executing %s %s [Timeout: %.1fs]",
            method,
            url,
            self._timeout,
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                status_code = resp.status
                raw_bytes = resp.read()
                raw_text = raw_bytes.decode("utf-8", errors="replace")

                try:
                    return json.loads(raw_text)
                except json.JSONDecodeError as exc:
                    logger.error("Failed to decode JSON from %s (HTTP %d): %s", url, status_code, exc)
                    raise InvalidJsonResponseError(
                        f"Server returned invalid JSON on {method} {path} (HTTP {status_code}): {raw_text[:200]}",
                        status_code=status_code,
                        response_body=raw_text,
                    ) from exc

        except urllib.error.HTTPError as exc:
            status_code = exc.code
            raw_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""

            logger.warning(
                "HTTP Error %d on %s %s: %s",
                status_code,
                method,
                path,
                raw_body[:300],
            )

            if status_code == 401:
                raise AuthenticationError(
                    f"HTTP 401 Unauthorized: Invalid or missing BOT_ACCESS_TOKEN on {method} {path}.",
                    status_code=status_code,
                    response_body=raw_body,
                ) from exc
            elif status_code == 403:
                raise AuthorizationError(
                    f"HTTP 403 Forbidden: Account verification or permission failure on {method} {path}: {raw_body}",
                    status_code=status_code,
                    response_body=raw_body,
                ) from exc
            elif status_code == 404:
                raise NotFoundError(
                    f"HTTP 404 Not Found: Endpoint or resource not found on {method} {path}.",
                    status_code=status_code,
                    response_body=raw_body,
                ) from exc
            elif status_code == 429:
                raise RateLimitError(
                    f"HTTP 429 Too Many Requests: Rate limit exceeded on {method} {path}.",
                    status_code=status_code,
                    response_body=raw_body,
                ) from exc
            elif status_code >= 500:
                raise ServerError(
                    f"HTTP {status_code} Server Error on {method} {path}: {raw_body}",
                    status_code=status_code,
                    response_body=raw_body,
                ) from exc
            else:
                raise TradingApiError(
                    f"HTTP {status_code} Error on {method} {path}: {raw_body}",
                    status_code=status_code,
                    response_body=raw_body,
                ) from exc

        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            # Check whether timeout occurred
            is_timeout = False
            if isinstance(exc, socket.timeout) or isinstance(exc, TimeoutError):
                is_timeout = True
            elif isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, socket.timeout):
                is_timeout = True
            elif "timed out" in str(exc).lower():
                is_timeout = True

            if is_timeout:
                logger.error("Request timed out after %.1fs on %s %s", self._timeout, method, path)
                raise RequestTimeoutError(
                    f"Request timed out after {self._timeout}s on {method} {path}. "
                    f"Outcome of state-changing operations is UNKNOWN."
                ) from exc

            logger.error("Network connection error on %s %s: %s", method, path, exc)
            raise NetworkError(f"Network error on {method} {path}: {exc}") from exc

    # ========================================================================
    @staticmethod
    def _safe_int(val: Any, default: int = 0) -> int:
        """Safely convert value to integer, defaulting if None or invalid."""
        if val is None:
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        """Safely convert value to float, defaulting if None or invalid."""
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # ========================================================================
    # Public Endpoint Methods
    # ========================================================================

    def get_stage(self) -> SimulationStage:
        """Fetch current platform simulation stage (GET /api/v1/stage)."""
        data = self._request("GET", "/api/v1/stage")
        if not isinstance(data, dict):
            raise InvalidJsonResponseError(f"Expected dict from /api/v1/stage, got {type(data).__name__}")

        tickers_list = list(data.get("tickers", [])) if isinstance(data.get("tickers"), list) else []
        return SimulationStage(
            stage=str(data.get("stage", "unknown")),
            status=str(data.get("status", "unknown")),
            tickers=tickers_list,
            ticker_count=self._safe_int(data.get("ticker_count"), len(tickers_list)),
            current_date=data.get("current_date"),
            days_completed=self._safe_int(data.get("days_completed"), 0),
            total_days=self._safe_int(data.get("total_days"), 0),
            interval_seconds=self._safe_int(data.get("interval_seconds"), 10),
            is_sandbox_stream=bool(data.get("is_sandbox_stream", False)),
            sandbox_loop=self._safe_int(data.get("sandbox_loop"), 1),
            raw_response=data,
        )

    def get_stocks(self) -> list[StockInfo]:
        """Fetch tradable stock universe (GET /api/v1/stocks)."""
        data = self._request("GET", "/api/v1/stocks")
        if not isinstance(data, dict):
            raise InvalidJsonResponseError(f"Expected dict from /api/v1/stocks, got {type(data).__name__}")

        raw_stocks = data.get("stocks", [])
        if not isinstance(raw_stocks, list):
            raise InvalidJsonResponseError(f"Expected 'stocks' list in /api/v1/stocks, got {type(raw_stocks).__name__}")

        result: list[StockInfo] = []
        for s in raw_stocks:
            if isinstance(s, dict):
                ticker = s.get("ticker") or s.get("symbol", "")
                result.append(
                    StockInfo(
                        symbol=str(s.get("symbol", ticker)),
                        ticker=str(ticker),
                        name=str(s.get("name", "")),
                        description=str(s.get("description", "")),
                    )
                )
        return result

    def get_stock_history(self, ticker: str, limit: int = 100) -> dict[str, Any]:
        """Fetch historical candle records for a stock (GET /api/v1/stocks/history)."""
        params = {"ticker": ticker, "limit": limit}
        return self._request("GET", "/api/v1/stocks/history", query_params=params)

    def get_simulation_clock(self) -> SimulationClock:
        """Fetch simulation clock details (GET /api/v1/simulation/clock)."""
        data = self._request("GET", "/api/v1/simulation/clock")
        if not isinstance(data, dict):
            raise InvalidJsonResponseError(f"Expected dict from /api/v1/simulation/clock, got {type(data).__name__}")

        return SimulationClock(
            status=str(data.get("status", "unknown")),
            simulated_date=str(data.get("simulated_date", "")),
            current_day=self._safe_int(data.get("current_day"), 0),
            days_completed=self._safe_int(data.get("days_completed"), 0),
            total_days=self._safe_int(data.get("total_days"), 0),
            remaining_days=self._safe_int(data.get("remaining_days"), 0),
            interval_seconds=self._safe_int(data.get("interval_seconds"), 10),
            raw_response=data,
        )

    def get_portfolio(self) -> Portfolio:
        """Fetch overall portfolio equity and holdings (GET /api/v1/portfolio)."""
        data = self._request("GET", "/api/v1/portfolio")
        if not isinstance(data, dict):
            raise InvalidJsonResponseError(f"Expected dict from /api/v1/portfolio, got {type(data).__name__}")

        holdings_list: list[PortfolioHolding] = []
        raw_holdings = data.get("holdings", [])
        if isinstance(raw_holdings, list):
            for h in raw_holdings:
                if isinstance(h, dict):
                    holdings_list.append(
                        PortfolioHolding(
                            ticker=str(h.get("ticker", "")),
                            quantity=self._safe_int(h.get("quantity"), 0),
                            avg_price=self._safe_float(h.get("avg_price", h.get("average_price")), 0.0),
                            current_price=self._safe_float(h.get("current_price"), 0.0),
                            market_value=self._safe_float(h.get("market_value"), 0.0),
                            unrealized_pnl=self._safe_float(h.get("unrealized_pnl"), 0.0),
                        )
                    )

        return Portfolio(
            team_id=str(data.get("team_id", "")),
            team_name=str(data.get("team_name", "")),
            available_cash=self._safe_float(data.get("available_cash"), 0.0),
            cash_balance=self._safe_float(data.get("cash_balance"), 0.0),
            portfolio_market_value=self._safe_float(data.get("portfolio_market_value"), 0.0),
            total_wealth=self._safe_float(data.get("total_wealth"), 0.0),
            realized_pnl=self._safe_float(data.get("realized_pnl"), 0.0),
            unrealized_pnl=self._safe_float(data.get("unrealized_pnl"), 0.0),
            total_pnl=self._safe_float(data.get("total_pnl"), 0.0),
            total_charges=self._safe_float(data.get("total_charges"), 0.0),
            total_taxes=self._safe_float(data.get("total_taxes"), 0.0),
            holdings=holdings_list,
            total_orders=self._safe_int(data.get("total_orders"), 0),
            rejected_orders_count=self._safe_int(data.get("rejected_orders_count"), 0),
            error_count=self._safe_int(data.get("error_count"), 0),
            is_blocked=bool(data.get("is_blocked", False)),
            is_verified=bool(data.get("is_verified", True)),
            simulated_date=data.get("simulated_date"),
            raw_response=data,
        )

    def get_cash(self) -> CashBalance:
        """Fetch cash and margin balance details (GET /api/v1/portfolio/cash)."""
        data = self._request("GET", "/api/v1/portfolio/cash")
        if not isinstance(data, dict):
            raise InvalidJsonResponseError(f"Expected dict from /api/v1/portfolio/cash, got {type(data).__name__}")

        return CashBalance(
            team_id=str(data.get("team_id", "")),
            team_name=str(data.get("team_name", "")),
            available_cash=self._safe_float(data.get("available_cash"), 0.0),
            cash_balance=self._safe_float(data.get("cash_balance"), 0.0),
            capital_remaining=self._safe_float(data.get("capital_remaining"), 0.0),
            margin_locked=self._safe_float(data.get("margin_locked"), 0.0),
            initial_capital=self._safe_float(data.get("initial_capital"), 0.0),
            realized_pnl=self._safe_float(data.get("realized_pnl"), 0.0),
            currency=str(data.get("currency", "INR")),
            is_blocked=bool(data.get("is_blocked", False)),
            simulated_date=data.get("simulated_date"),
            raw_response=data,
        )

    get_portfolio_cash = get_cash

    def get_orders(
        self,
        page: int = 1,
        limit: int = 50,
        status: Optional[str] = None,
        action: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> dict[str, Any]:
        """Fetch historical orders with pagination and filtering (GET /api/v1/orders)."""
        params: dict[str, Any] = {
            "page": page,
            "limit": limit,
            "status": status,
            "action": action,
            "ticker": ticker,
        }
        return self._request("GET", "/api/v1/orders", query_params=params)

    def submit_order(
        self,
        ticker: str,
        action: str,
        quantity: int,
        date: Optional[str] = None,
    ) -> dict[str, Any]:
        """Submit a market order to the competition platform (POST /api/v1/orders).

        Args:
            ticker: Asset ticker symbol.
            action: Documented action ('BUY', 'SELL', 'SHORT', 'COVER').
            quantity: Positive integer quantity.
            date: Optional simulated market date.

        Returns:
            Server JSON response dictionary.
        """
        payload: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
        }
        if date:
            payload["date"] = date

        return self._request("POST", "/api/v1/orders", payload=payload)

    def send_heartbeat(self, seq: int | str) -> dict[str, Any]:
        """Send preflight heartbeat acknowledgement (POST /api/v1/bot/heartbeat).

        Args:
            seq: Sequence number from preflight tick.

        Returns:
            Server JSON response dictionary.
        """
        payload = {"seq": str(seq)}
        return self._request("POST", "/api/v1/bot/heartbeat", payload=payload)
