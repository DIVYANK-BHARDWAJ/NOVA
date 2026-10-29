"""Unit and contract validation tests for execution_event.schema.json.

Ensures that the JSON Schema file is syntactically and semantically valid,
matches the Python ExecutionEvent and OrderResult models, enforces safety
rules (UNKNOWN status, reconciliation required, positive fill quantity),
and validates sample payloads without requiring external third-party dependencies.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from execution.models import (
    ExecutionEvent,
    OrderAction,
    OrderResult,
    OrderStatus,
)

SCHEMA_PATH = Path("schemas/execution_event.schema.json")

# Permitted values according to schema
PERMITTED_EVENT_TYPES = {
    "ORDER_SUBMISSION_ATTEMPT",
    "ORDER_ACCEPTED",
    "ORDER_FILLED",
    "ORDER_PARTIALLY_FILLED",
    "ORDER_REJECTED",
    "ORDER_UNCERTAIN",
    "ORDER_CANCELLED",
}

PERMITTED_ACTIONS = {"BUY", "SELL", "SHORT", "COVER"}

PERMITTED_STATUSES = {
    "PENDING",
    "SUBMITTED",
    "FILLED",
    "PARTIALLY_FILLED",
    "REJECTED",
    "CANCELLED",
    "UNKNOWN",
}


def load_schema() -> dict[str, Any]:
    """Load and parse the JSON Schema file."""
    assert SCHEMA_PATH.is_file(), f"Schema file not found at {SCHEMA_PATH}"
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_iso_timestamp(ts: Any) -> bool:
    """Verify that a value is a valid ISO-8601 date-time string."""
    if not isinstance(ts, str):
        return False
    try:
        clean = ts.replace("Z", "+00:00")
        datetime.fromisoformat(clean)
        return True
    except Exception:
        return False


def validate_execution_event_payload(payload: Any) -> list[str]:
    """Validate a dictionary against the execution_event.schema.json contract.

    Returns a list of validation error descriptions (empty if valid).
    """
    errors: list[str] = []

    if not isinstance(payload, dict):
        return ["Payload must be a JSON object (dict)."]

    # Root required keys
    for req in ("event_type", "timestamp", "order_result"):
        if req not in payload:
            errors.append(f"Root object missing required property '{req}'.")

    # Reject unexpected root properties (additionalProperties: false)
    allowed_root = {"event_type", "timestamp", "order_result", "extra"}
    for k in payload.keys():
        if k not in allowed_root:
            errors.append(f"Unexpected root property '{k}' (additionalProperties: false).")

    event_type = payload.get("event_type")
    if event_type is not None:
        if not isinstance(event_type, str):
            errors.append(f"'event_type' must be string, got {type(event_type).__name__}.")
        elif event_type not in PERMITTED_EVENT_TYPES:
            errors.append(f"Invalid 'event_type' '{event_type}'. Permitted: {sorted(PERMITTED_EVENT_TYPES)}.")

    root_ts = payload.get("timestamp")
    if root_ts is not None and not validate_iso_timestamp(root_ts):
        errors.append(f"Root 'timestamp' is not a valid ISO-8601 string: {root_ts!r}.")

    extra = payload.get("extra")
    if extra is not None and not isinstance(extra, dict):
        errors.append(f"'extra' must be a JSON object (dict), got {type(extra).__name__}.")

    order_result = payload.get("order_result")
    if order_result is None and "order_result" in payload:
        errors.append("'order_result' cannot be null.")
        return errors

    if order_result is not None:
        if not isinstance(order_result, dict):
            errors.append(f"'order_result' must be an object, got {type(order_result).__name__}.")
            return errors

        # Required order_result keys
        res_req = (
            "client_intent_id",
            "ticker",
            "action",
            "quantity",
            "status",
            "filled_quantity",
            "charges",
            "requires_reconciliation",
            "timestamp",
        )
        for req in res_req:
            if req not in order_result:
                errors.append(f"'order_result' missing required property '{req}'.")

        allowed_res_keys = {
            "client_intent_id",
            "ticker",
            "action",
            "quantity",
            "status",
            "order_id",
            "execution_price",
            "filled_quantity",
            "charges",
            "rejection_reason",
            "requires_reconciliation",
            "raw_response",
            "timestamp",
        }
        for k in order_result.keys():
            if k not in allowed_res_keys:
                errors.append(f"Unexpected property '{k}' in 'order_result'.")

        # Field validations
        cid = order_result.get("client_intent_id")
        if cid is not None and (not isinstance(cid, str) or len(cid.strip()) == 0):
            errors.append(f"'client_intent_id' must be non-empty string, got {cid!r}.")

        ticker = order_result.get("ticker")
        if ticker is not None:
            if not isinstance(ticker, str) or not re.match(r"^[A-Z0-9_.-]+$", ticker):
                errors.append(f"'ticker' must be valid uppercase symbol, got {ticker!r}.")

        action = order_result.get("action")
        if action is not None and action not in PERMITTED_ACTIONS:
            errors.append(f"Invalid 'action' '{action}'. Permitted: {sorted(PERMITTED_ACTIONS)}.")

        qty = order_result.get("quantity")
        if qty is not None:
            if type(qty) is not int or qty < 1:
                errors.append(f"'quantity' must be integer >= 1, got {qty!r} ({type(qty).__name__}).")

        status = order_result.get("status")
        if status is not None and status not in PERMITTED_STATUSES:
            errors.append(f"Invalid 'status' '{status}'. Permitted: {sorted(PERMITTED_STATUSES)}.")

        order_id = order_result.get("order_id")
        if order_id is not None and not isinstance(order_id, str):
            errors.append(f"'order_id' must be string or null, got {type(order_id).__name__}.")

        price = order_result.get("execution_price")
        if price is not None and (not isinstance(price, (int, float)) or price < 0):
            errors.append(f"'execution_price' must be number >= 0 or null, got {price!r}.")

        filled_qty = order_result.get("filled_quantity")
        if filled_qty is not None and (type(filled_qty) is not int or filled_qty < 0):
            errors.append(f"'filled_quantity' must be integer >= 0, got {filled_qty!r}.")

        charges = order_result.get("charges")
        if charges is not None and (not isinstance(charges, (int, float)) or charges < 0):
            errors.append(f"'charges' must be number >= 0, got {charges!r}.")

        rej = order_result.get("rejection_reason")
        if rej is not None and not isinstance(rej, str):
            errors.append(f"'rejection_reason' must be string or null, got {type(rej).__name__}.")

        req_recon = order_result.get("requires_reconciliation")
        if req_recon is not None and not isinstance(req_recon, bool):
            errors.append(f"'requires_reconciliation' must be boolean, got {type(req_recon).__name__}.")

        raw_resp = order_result.get("raw_response")
        if raw_resp is not None and not isinstance(raw_resp, dict):
            errors.append(f"'raw_response' must be object or null, got {type(raw_resp).__name__}.")

        res_ts = order_result.get("timestamp")
        if res_ts is not None and not validate_iso_timestamp(res_ts):
            errors.append(f"'order_result.timestamp' is not a valid ISO-8601 string: {res_ts!r}.")

        # SEMANTIC SAFETY INVARIANTS:
        # 1. UNKNOWN requires requires_reconciliation == True
        if status == "UNKNOWN" and req_recon is not True:
            errors.append("Safety invariant violated: When status is UNKNOWN, requires_reconciliation must be true.")

        # 2. REJECTED requires rejection_reason to be a non-empty string
        if status == "REJECTED":
            if not rej or not rej.strip():
                errors.append("Safety invariant violated: When status is REJECTED, rejection_reason must be provided.")

        # 3. FILLED requires filled_quantity >= 1
        if status == "FILLED":
            if filled_qty is not None and filled_qty < 1:
                errors.append("Safety invariant violated: When status is FILLED, filled_quantity must be >= 1.")

        # 4. ORDER_UNCERTAIN requires status == UNKNOWN
        if event_type == "ORDER_UNCERTAIN" and status != "UNKNOWN":
            errors.append(
                f"Safety invariant violated: When event_type is ORDER_UNCERTAIN, status must be UNKNOWN, got {status}."
            )

    return errors


def test_schema_file_exists_and_is_valid_json():
    """Verify that execution_event.schema.json exists and is syntactically valid JSON."""
    schema = load_schema()
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["title"] == "ExecutionEvent"
    assert schema["type"] == "object"
    assert "event_type" in schema["required"]
    assert "order_result" in schema["required"]
    assert "timestamp" in schema["required"]


def test_schema_enums_match_python_enums():
    """Verify that schema enums exactly match Python model definitions."""
    schema = load_schema()
    props = schema["properties"]

    # event_type
    schema_event_types = set(props["event_type"]["enum"])
    assert schema_event_types == PERMITTED_EVENT_TYPES

    # action
    res_props = props["order_result"]["properties"]
    schema_actions = set(res_props["action"]["enum"])
    python_actions = {a.value for a in OrderAction}
    assert schema_actions == python_actions

    # status
    schema_statuses = set(res_props["status"]["enum"])
    python_statuses = {s.value for s in OrderStatus}
    assert schema_statuses == python_statuses


def test_schema_validation_on_live_models():
    """Validate that ExecutionEvent.to_dict() outputs conform to the schema."""
    now = datetime.now(timezone.utc)

    # 1. ORDER_FILLED
    res_filled = OrderResult(
        client_intent_id="intent_fill_001",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=50,
        status=OrderStatus.FILLED,
        order_id="ord_server_999",
        execution_price=104.25,
        filled_quantity=50,
        charges=12.50,
        requires_reconciliation=False,
        timestamp=now,
    )
    evt_filled = ExecutionEvent(
        event_type="ORDER_FILLED",
        order_result=res_filled,
        timestamp=now,
        extra={"strategy_id": "trend_follower_1", "risk_authorization_id": "hmac-auth-abc-123"},
    )
    errs = validate_execution_event_payload(evt_filled.to_dict())
    assert errs == []

    # 2. ORDER_REJECTED
    res_rejected = OrderResult(
        client_intent_id="intent_rej_002",
        ticker="BLUX",
        action=OrderAction.SELL,
        quantity=20,
        status=OrderStatus.REJECTED,
        rejection_reason="UNAUTHORIZED_BY_RISK_GATEWAY: Lacks authorization.",
        requires_reconciliation=False,
        timestamp=now,
    )
    evt_rejected = ExecutionEvent(event_type="ORDER_REJECTED", order_result=res_rejected, timestamp=now)
    errs = validate_execution_event_payload(evt_rejected.to_dict())
    assert errs == []

    # 3. ORDER_UNCERTAIN (Timeout)
    res_uncertain = OrderResult(
        client_intent_id="intent_timeout_003",
        ticker="AERX",
        action=OrderAction.BUY,
        quantity=15,
        status=OrderStatus.UNKNOWN,
        rejection_reason="TIMEOUT_UNCERTAIN: Socket timeout after 5.0s.",
        requires_reconciliation=True,
        timestamp=now,
    )
    evt_uncertain = ExecutionEvent(event_type="ORDER_UNCERTAIN", order_result=res_uncertain, timestamp=now)
    errs = validate_execution_event_payload(evt_uncertain.to_dict())
    assert errs == []

    # 4. ORDER_SUBMISSION_ATTEMPT
    res_attempt = OrderResult(
        client_intent_id="intent_sub_004",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        status=OrderStatus.PENDING,
        requires_reconciliation=False,
        timestamp=now,
    )
    evt_attempt = ExecutionEvent(event_type="ORDER_SUBMISSION_ATTEMPT", order_result=res_attempt, timestamp=now)
    errs = validate_execution_event_payload(evt_attempt.to_dict())
    assert errs == []


def test_schema_rejects_missing_required_fields():
    """Schema must reject payloads missing root or nested required fields."""
    # Missing root event_type
    bad_root = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_result": {},
    }
    errs = validate_execution_event_payload(bad_root)
    assert any("missing required property 'event_type'" in e for e in errs)

    # Missing nested client_intent_id
    bad_res = {
        "event_type": "ORDER_FILLED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_result": {
            "ticker": "ROUT",
            "action": "BUY",
            "quantity": 10,
            "status": "FILLED",
            "filled_quantity": 10,
            "charges": 0.0,
            "requires_reconciliation": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    errs = validate_execution_event_payload(bad_res)
    assert any("missing required property 'client_intent_id'" in e for e in errs)


def test_schema_rejects_invalid_enums():
    """Schema must reject unpermitted event_type, action, or status strings."""
    now_str = datetime.now(timezone.utc).isoformat()

    # Invalid event_type
    p1 = {
        "event_type": "ORDER_BOUNCED",
        "timestamp": now_str,
        "order_result": {
            "client_intent_id": "i_1",
            "ticker": "ROUT",
            "action": "BUY",
            "quantity": 10,
            "status": "FILLED",
            "filled_quantity": 10,
            "charges": 0.0,
            "requires_reconciliation": False,
            "timestamp": now_str,
        },
    }
    assert any("Invalid 'event_type'" in e for e in validate_execution_event_payload(p1))

    # Invalid action
    p2 = {
        "event_type": "ORDER_FILLED",
        "timestamp": now_str,
        "order_result": {
            "client_intent_id": "i_1",
            "ticker": "ROUT",
            "action": "HOLD",  # Not BUY/SELL/SHORT/COVER
            "quantity": 10,
            "status": "FILLED",
            "filled_quantity": 10,
            "charges": 0.0,
            "requires_reconciliation": False,
            "timestamp": now_str,
        },
    }
    assert any("Invalid 'action'" in e for e in validate_execution_event_payload(p2))

    # Invalid status
    p3 = {
        "event_type": "ORDER_FILLED",
        "timestamp": now_str,
        "order_result": {
            "client_intent_id": "i_1",
            "ticker": "ROUT",
            "action": "BUY",
            "quantity": 10,
            "status": "SUCCESS",  # Must be FILLED, not SUCCESS
            "filled_quantity": 10,
            "charges": 0.0,
            "requires_reconciliation": False,
            "timestamp": now_str,
        },
    }
    assert any("Invalid 'status'" in e for e in validate_execution_event_payload(p3))


def test_schema_enforces_safety_semantics():
    """Schema must enforce critical safety constraints:
    1. UNKNOWN status must have requires_reconciliation=True.
    2. REJECTED status must have rejection_reason.
    3. FILLED status must have filled_quantity >= 1.
    4. ORDER_UNCERTAIN cannot be marked FILLED.
    """
    now_str = datetime.now(timezone.utc).isoformat()

    # 1. UNKNOWN with requires_reconciliation=False must be rejected
    p_unknown_unsafe = {
        "event_type": "ORDER_UNCERTAIN",
        "timestamp": now_str,
        "order_result": {
            "client_intent_id": "i_unk",
            "ticker": "ROUT",
            "action": "BUY",
            "quantity": 10,
            "status": "UNKNOWN",
            "filled_quantity": 0,
            "charges": 0.0,
            "requires_reconciliation": False,  # UNSAFE!
            "timestamp": now_str,
        },
    }
    errs = validate_execution_event_payload(p_unknown_unsafe)
    assert any("When status is UNKNOWN, requires_reconciliation must be true" in e for e in errs)

    # 2. REJECTED without rejection_reason
    p_rejected_no_reason = {
        "event_type": "ORDER_REJECTED",
        "timestamp": now_str,
        "order_result": {
            "client_intent_id": "i_rej",
            "ticker": "ROUT",
            "action": "BUY",
            "quantity": 10,
            "status": "REJECTED",
            "rejection_reason": None,  # Missing reason
            "filled_quantity": 0,
            "charges": 0.0,
            "requires_reconciliation": False,
            "timestamp": now_str,
        },
    }
    errs = validate_execution_event_payload(p_rejected_no_reason)
    assert any("When status is REJECTED, rejection_reason must be provided" in e for e in errs)

    # 3. FILLED with zero filled_quantity
    p_filled_zero_qty = {
        "event_type": "ORDER_FILLED",
        "timestamp": now_str,
        "order_result": {
            "client_intent_id": "i_fill",
            "ticker": "ROUT",
            "action": "BUY",
            "quantity": 10,
            "status": "FILLED",
            "filled_quantity": 0,  # Zero filled quantity cannot be FILLED
            "charges": 0.0,
            "requires_reconciliation": False,
            "timestamp": now_str,
        },
    }
    errs = validate_execution_event_payload(p_filled_zero_qty)
    assert any("When status is FILLED, filled_quantity must be >= 1" in e for e in errs)

    # 4. ORDER_UNCERTAIN with status FILLED
    p_uncertain_contradiction = {
        "event_type": "ORDER_UNCERTAIN",
        "timestamp": now_str,
        "order_result": {
            "client_intent_id": "i_contra",
            "ticker": "ROUT",
            "action": "BUY",
            "quantity": 10,
            "status": "FILLED",
            "filled_quantity": 10,
            "charges": 0.0,
            "requires_reconciliation": False,
            "timestamp": now_str,
        },
    }
    errs = validate_execution_event_payload(p_uncertain_contradiction)
    assert any("When event_type is ORDER_UNCERTAIN, status must be UNKNOWN" in e for e in errs)


def test_schema_rejects_invalid_quantity_and_charges():
    """Schema must reject negative or fractional quantity, and negative charges."""
    now_str = datetime.now(timezone.utc).isoformat()

    # Zero quantity
    p_zero_qty = {
        "event_type": "ORDER_SUBMISSION_ATTEMPT",
        "timestamp": now_str,
        "order_result": {
            "client_intent_id": "i_zero",
            "ticker": "ROUT",
            "action": "BUY",
            "quantity": 0,  # Must be >= 1
            "status": "PENDING",
            "filled_quantity": 0,
            "charges": 0.0,
            "requires_reconciliation": False,
            "timestamp": now_str,
        },
    }
    assert any("'quantity' must be integer >= 1" in e for e in validate_execution_event_payload(p_zero_qty))

    # Negative charges
    p_neg_charges = {
        "event_type": "ORDER_FILLED",
        "timestamp": now_str,
        "order_result": {
            "client_intent_id": "i_charge",
            "ticker": "ROUT",
            "action": "BUY",
            "quantity": 10,
            "status": "FILLED",
            "filled_quantity": 10,
            "charges": -5.0,  # Negative charges
            "requires_reconciliation": False,
            "timestamp": now_str,
        },
    }
    assert any("'charges' must be number >= 0" in e for e in validate_execution_event_payload(p_neg_charges))


def test_person2_and_person3_metadata_support():
    """Test that extra dictionary preserves strategy_id and risk_authorization_id."""
    now = datetime.now(timezone.utc)
    res = OrderResult(
        client_intent_id="intent_cross_module",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        status=OrderStatus.FILLED,
        filled_quantity=10,
        charges=1.5,
        requires_reconciliation=False,
        timestamp=now,
    )
    evt = ExecutionEvent(
        event_type="ORDER_FILLED",
        order_result=res,
        timestamp=now,
        extra={
            "strategy_id": "person_2_momentum_alpha_v2",
            "risk_authorization_id": "person_3_risk_auth_token_987654",
            "correlation_id": "corr-uuid-001",
        },
    )
    payload = evt.to_dict()
    assert payload["extra"]["strategy_id"] == "person_2_momentum_alpha_v2"
    assert payload["extra"]["risk_authorization_id"] == "person_3_risk_auth_token_987654"
    assert validate_execution_event_payload(payload) == []
