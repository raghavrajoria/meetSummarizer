# ASR service contract

The remote provider sends one recording to the configured `ASR_SERVICE_URL` as
an HTTP `POST` with `multipart/form-data`. The audio file is the `audio` form
field. When `ASR_SERVICE_TOKEN` is configured, the client sends it as a bearer
token. The client timeout is configured by `ASR_SERVICE_TIMEOUT_SECONDS`.

The service responds with HTTP 200 and a JSON array. Every array item must have:

| Field | Type / accepted values |
|---|---|
| `idx` | non-negative integer |
| `start`, `end` | finite numbers in seconds; end must be at or after start |
| `speaker` | non-empty string |
| `text` | string |
| `lang`, `method` | non-empty strings |
| `quality` | `accepted`, `review`, or `rejected` |
| `reasons` | array of strings |
| `duration` | finite non-negative number in seconds |
| `confidence` | number or `null` |
| `words` | optional array of objects |

Additional row fields are retained. `words` object fields and the maximum audio
length are unknown and are not constrained by this client. The client validates
the whole response and reports row/field validation errors without including
transcript text in the error message.

The import provider reads a local `asr.json` file containing the same array.
