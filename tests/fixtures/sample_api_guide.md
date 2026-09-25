# Sample API Guide — CMN-C1-053 Test Fixture (Updated v2)

This guide covers the Sample Commerce API used in CMN-C1-053 test fixtures.
It serves as a comprehensive API documentation example covering authentication,
rate limiting, pagination, error handling, and webhook integration.

---

## Overview

The Sample Commerce API is a RESTful API that provides order management,
catalog browsing, authentication, and webhook notification capabilities.
All requests must be authenticated with a valid Bearer token (except the auth endpoint).

**Base URL:** `https://api.example.com`  
**API Version:** `1.2.0`  
**Protocol:** HTTPS only. HTTP connections are rejected with `301 Moved Permanently`.

---

## Authentication

### Obtaining a Bearer Token

Authenticate by sending a `POST /v1/auth` request with your credentials:

```http
POST /v1/auth HTTP/1.1
Host: api.example.com
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "your-password"
}
```

A successful `200 OK` response returns your Bearer token and expiry:

```json
{
  "token": "<your-bearer-token>",
  "expires_at": "2026-05-29T10:00:00Z",
  "token_type": "Bearer"
}
```

**Token lifetime:** 3600 seconds (1 hour). Request a new token before expiry.

### Using Your Token

Include the token in the `Authorization` header of every authenticated request:

```http
GET /v1/orders HTTP/1.1
Host: api.example.com
Authorization: Bearer <your-token-here>
```

### Token Expiry

When your token expires, you receive a `401 Unauthorized` response with:

```json
{
  "error": "token_expired",
  "message": "Your token has expired. Please re-authenticate."
}
```

Re-authenticate using `POST /v1/auth` to obtain a fresh token.

### Authentication Error Codes

| HTTP Status | Error Code | Meaning |
|---|---|---|
| `400 Bad Request` | `malformed_request` | Missing username or password field |
| `401 Unauthorized` | `invalid_credentials` | Username or password incorrect |
| `401 Unauthorized` | `token_expired` | Bearer token has expired |
| `403 Forbidden` | `insufficient_permissions` | Token valid but lacks required scope |
| `429 Too Many Requests` | `rate_limit_exceeded` | Too many auth attempts; check `Retry-After` header |

---

## Rate Limiting

### Limits

| Endpoint Group | Limit | Window |
|---|---|---|
| `POST /v1/auth` | 10 requests | per minute per IP |
| All other endpoints | 500 requests | per minute per token |
| Webhook registrations | 50 registrations | per account |

### Rate Limit Headers

Every API response includes rate limit headers:

```
X-RateLimit-Limit: 500
X-RateLimit-Remaining: 487
X-RateLimit-Reset: 1748516400
```

- `X-RateLimit-Limit`: Maximum requests allowed in the current window
- `X-RateLimit-Remaining`: Requests remaining in the current window
- `X-RateLimit-Reset`: Unix timestamp when the limit resets

### Handling 429 Responses

When you exceed the rate limit, you receive `429 Too Many Requests`:

```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Please wait before retrying.",
  "request_id": "req_abc123"
}
```

Check the `Retry-After` header for the number of seconds to wait:

```http
Retry-After: 37
```

**Best practice:** Implement exponential backoff starting at the `Retry-After` value.

---

## Pagination

All list endpoints (`GET /v1/orders`, `GET /v1/items`, `GET /v1/webhooks`) support
offset-based pagination using `limit` and `offset` query parameters.

### Parameters

| Parameter | Type | Default | Maximum | Description |
|---|---|---|---|---|
| `limit` | integer | 20 | 100 | Results per page |
| `offset` | integer | 0 | — | Starting position |

### Response Shape

Every paginated response includes `total`, `limit`, and `offset`:

```json
{
  "items": [...],
  "total": 157,
  "limit": 20,
  "offset": 0
}
```

### Iterating All Pages

```python
offset = 0
limit = 100
all_orders = []

while True:
    resp = get("/v1/orders", params={"limit": limit, "offset": offset})
    all_orders.extend(resp["items"])
    if offset + limit >= resp["total"]:
        break
    offset += limit
```

---

## Error Handling

### Error Response Format

All errors return a consistent JSON body:

```json
{
  "error": "<machine_readable_code>",
  "message": "<human_readable_description>",
  "request_id": "<trace_id_for_support>"
}
```

Always check the HTTP status code first, then the `error` field for
machine-readable classification.

### Common Error Codes

| HTTP Status | Error Code | Description |
|---|---|---|
| `400` | `malformed_request` | Request body is not valid JSON or missing required fields |
| `401` | `invalid_credentials` | Authentication failed |
| `401` | `token_expired` | Bearer token has expired |
| `403` | `insufficient_permissions` | Action not allowed for this account |
| `404` | `not_found` | Requested resource does not exist |
| `422` | `validation_error` | Input data failed validation (see `details` array) |
| `429` | `rate_limit_exceeded` | Too many requests; wait for `Retry-After` |
| `500` | `internal_error` | Server-side error; retry with backoff |
| `503` | `service_unavailable` | Temporary outage; retry with backoff |

### Validation Error Details

`422` responses include a `details` array identifying which fields failed:

```json
{
  "error": "validation_error",
  "message": "One or more fields are invalid.",
  "details": [
    { "field": "items[0].quantity", "issue": "Must be >= 1" },
    { "field": "items[1].product_id", "issue": "Unknown product ID" }
  ]
}
```

---

## Webhook Integration

### Registering a Webhook

Subscribe to order and item events by registering a webhook endpoint:

```http
POST /v1/webhooks HTTP/1.1
Authorization: Bearer <token>
Content-Type: application/json

{
  "url": "https://your-server.example.com/webhooks/orders",
  "events": ["order.created", "order.fulfilled", "order.cancelled"],
  "secret": "optional-hmac-signing-secret"
}
```

Supported event types: `order.created`, `order.fulfilled`, `order.cancelled`, `item.updated`.

### Webhook Payload

Every webhook delivery is an HTTP `POST` to your registered URL:

```json
{
  "event": "order.created",
  "timestamp": "2026-05-29T09:00:00Z",
  "data": {
    "order_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "pending"
  }
}
```

### Verifying Webhook Signatures

If you provided a `secret` at registration, the payload includes an
`X-Signature-SHA256` header:

```
X-Signature-SHA256: sha256=<hmac_hex_digest>
```

Verify the signature before processing the webhook:

```python
import hmac, hashlib

def verify(payload_bytes: bytes, header_sig: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, header_sig)
```

### Webhook Retry Policy

Failed webhook deliveries (non-2xx response or timeout) are retried
with exponential backoff: 1 min, 5 min, 30 min, 2 hr, 8 hr (5 attempts total).
After all retries fail, the webhook is marked as `failed` and no further
deliveries are attempted until you re-enable it.
