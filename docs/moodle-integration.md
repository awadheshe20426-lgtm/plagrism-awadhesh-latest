# Moodle Integration Guide — Plagiarism Detection API

**Audience:** a Moodle developer building a `plagiarism_*` plugin that sends
Assignment submissions to this Plagiarism Detection service and shows the results
in the grading view.

**API base URL:** `http://35.154.140.122:8001` (the frontend host proxies
`/api/**`, `/docs`, `/openapi.json` to the backend). Use HTTPS in production.

**Live API reference (Swagger):** `http://35.154.140.122:8001/docs`
**OpenAPI spec:** `http://35.154.140.122:8001/openapi.json`

> All examples below are copy-pasteable `curl`. Replace `$BASE` with the base URL,
> `$TOKEN` with a bearer token (see Authentication).

---

## 1. How the integration works (the flow)

```
Student submits a file in a Moodle Assignment
        │  event: \assignsubmission_file\event\assessable_uploaded
        ▼
[Moodle plugin] observer  ──queues──►  adhoc task   (never block the submit!)
        │
        ▼  POST /api/v1/analyze  (multipart: file + options)   ── returns batch_id
        │
        ▼  scheduled task polls  GET /api/v1/batches/{batch_id}/status  until done
        │
        ▼  GET /api/v1/batches/{batch_id}/results     ── per-document scores + matches
        │
        ▼  store score locally; get_links() renders a badge + report link in grader view
```

Key rule (Moodle plugin best practice): **do the HTTP calls in a queued task, not
in the event handler** — a student's submit must never wait on this API.

---

## 2. Authentication

The API uses **JWT bearer tokens** (fastapi-users). For a server-to-server Moodle
plugin, create a dedicated **service account** and log in to obtain a token.

- Token lifetime: **3600 s (1 hour)** → cache it and refresh on expiry / on `401`.
- Header for every call: `Authorization: Bearer <access_token>`.

### 2.1 Create the service account (once)

```bash
curl -s -X POST "$BASE/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email":"moodle-service@yourschool.edu","password":"CHANGE_ME_STRONG"}'
```

(An admin can also create it via `POST /api/admin/users`. Faculty/admin role is
needed to create assignments and to see cross-batch matches from other owners.)

### 2.2 Log in → get a token

`POST /api/v1/auth/jwt/login` — **form-encoded** (`username` = the email):

```bash
curl -s -X POST "$BASE/api/v1/auth/jwt/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=moodle-service@yourschool.edu" \
  --data-urlencode "password=CHANGE_ME_STRONG"
# → {"access_token":"eyJhbGciOi...","token_type":"bearer"}
```

Verify a token any time: `GET /api/v1/users/me` (200 = valid, 401 = expired).

---

## 3. (Optional) Group submissions under an Assignment

Creating an **Assignment** lets the service scope cross-batch comparison and carry
per-activity thresholds. Map it to a Moodle **course module id (cmid)** via
`external_ref`.

```bash
curl -s -X POST "$BASE/api/v1/assignments" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
        "name": "PHIL101 Essay 2",
        "external_ref": "cmid-4821",
        "chunk_threshold": 0.8,
        "doc_threshold": 0.3,
        "strip_quotes": true,
        "strip_references": true,
        "boilerplate_text": null
      }'
# → {"assignment_id":"<uuid>","name":"PHIL101 Essay 2"}
```

Cache the returned `assignment_id` keyed by cmid in your plugin's DB. List existing
ones with `GET /api/v1/assignments`.

---

## 4. Submit a document for analysis

`POST /api/v1/analyze` — **multipart/form-data**. Send one file per student
submission (or up to `MAX_FILES_PER_BATCH = 50` as one batch).

Fields:
| field | type | notes |
|---|---|---|
| `files` | file(s) | `.pdf .docx .txt .png .jpg .jpeg`, ≤ **25 MB** each |
| `text` | string | alternative to `files` (online-text submissions) |
| `options` | JSON string | see below |

`options` JSON:
```json
{
  "provider": "local",
  "ai_threshold": 0.5,
  "check_plagiarism": true,
  "check_ai": true,
  "assignment_id": "<uuid or omit>"
}
```

```bash
curl -s -X POST "$BASE/api/v1/analyze" \
  -H "Authorization: Bearer $TOKEN" \
  -F "files=@submission.pdf" \
  -F 'options={"provider":"local","check_plagiarism":true,"check_ai":true,"assignment_id":"<uuid>"}'
# → {"batch_id":"<uuid>","status":"queued","message":"..."}
```

Store the returned **`batch_id`** against the Moodle submission (cmid + userid +
file). Processing is asynchronous (a worker embeds, fingerprints and compares).

---

## 5. Poll for completion

`GET /api/v1/batches/{batch_id}/status` — cheap; safe to poll every few seconds
from your scheduled task.

```bash
curl -s "$BASE/api/v1/batches/<batch_id>/status" -H "Authorization: Bearer $TOKEN"
# → {"batch_id":"...","status":"processing","total":1,"completed":0,"failed":0,
#     "processing":1,"queued":0,"percent":0,"done":false}
```

Poll until `"done": true` (status `completed` or `failed`).

---

## 6. Fetch results

`GET /api/v1/batches/{batch_id}/results` → a list, one object per document:

```json
[
  {
    "document_id": "<uuid>",
    "filename": "submission.pdf",
    "status": "completed",
    "ocr": { "used": false, "confidence": null, "low_confidence": false },
    "ai_analysis": {
      "score": 0.12, "is_ai": false, "confidence": 0.9, "provider": "local",
      "advisory": true,
      "note": "AI-detection scores are indicative only ... not sole proof."
    },
    "plagiarism_analysis": [
      {
        "similar_document": "otherstudent.pdf",
        "similarity": 0.83,
        "matches": [ /* chunk-level spans for highlighting */ ],
        "source_batch_id": "<uuid|null>",
        "source_owner": "prof@school.edu",
        "source_uploaded_at": "2026-07-10T09:00:00Z",
        "cross_batch": true
      }
    ]
  }
]
```

Take the **top `similarity`** in `plagiarism_analysis` as the headline % for the
badge. Treat `ai_analysis` as **advisory only** (never as proof — see the `note`).

Exports for the grader: `GET /api/v1/batches/{batch_id}/report.csv` and
`.../report.pdf`.

---

## 7. Building the Moodle plugin (`plagiarism_yourbrand`)

Standard Moodle Plagiarism API layout. Only the pieces that touch this API are
shown; see any core plagiarism plugin for the rest.

### 7.1 Config (admin settings.php)
- `api_base_url` (e.g. `https://…:8001`)
- `service_email`, `service_password` (the service account)
- default thresholds / scope

### 7.2 API client (`classes/api_client.php`) — JWT login + token cache
```php
namespace plagiarism_yourbrand;
class api_client {
    private $base;
    public function __construct() {
        $this->base = rtrim(get_config('plagiarism_yourbrand', 'api_base_url'), '/');
    }
    /** Cached bearer token (refresh a little before the 1h expiry). */
    private function token(): string {
        $cache = \cache::make('plagiarism_yourbrand', 'token');
        if ($t = $cache->get('bearer')) { return $t; }
        require_once($GLOBALS['CFG']->libdir.'/filelib.php');
        $curl = new \curl();
        $curl->setHeader('Content-Type: application/x-www-form-urlencoded');
        $resp = $curl->post($this->base.'/api/v1/auth/jwt/login', http_build_query([
            'username' => get_config('plagiarism_yourbrand', 'service_email'),
            'password' => get_config('plagiarism_yourbrand', 'service_password'),
        ]));
        $data = json_decode($resp, true);
        if (empty($data['access_token'])) { throw new \moodle_exception('authfailed', 'plagiarism_yourbrand'); }
        $cache->set('bearer', $data['access_token']);   // TTL ~3300s in cache definition
        return $data['access_token'];
    }
    private function curl(): \curl {
        $c = new \curl();
        $c->setHeader('Authorization: Bearer '.$this->token());
        return $c;
    }
    public function analyze(\stored_file $file, ?string $assignmentid): array {
        $tmp = make_request_directory().'/'.$file->get_filename();
        $file->copy_content_to($tmp);
        $opts = json_encode(['provider'=>'local','check_plagiarism'=>true,'check_ai'=>true,'assignment_id'=>$assignmentid]);
        $resp = $this->curl()->post($this->base.'/api/v1/analyze', [
            'files'   => new \CURLFile($tmp, $file->get_mimetype(), $file->get_filename()),
            'options' => $opts,
        ]);
        return json_decode($resp, true);      // ['batch_id'=>..,'status'=>..]
    }
    public function status(string $batchid): array {
        return json_decode($this->curl()->get($this->base."/api/v1/batches/$batchid/status"), true);
    }
    public function results(string $batchid): array {
        return json_decode($this->curl()->get($this->base."/api/v1/batches/$batchid/results"), true);
    }
}
```
On any `401`, clear the cached token and retry once (token expired).

### 7.3 Observer → queue only (`classes/observer.php`)
```php
public static function assessable_uploaded(\core\event\base $event) {
    if (!self::enabled_for_cm($event->contextinstanceid)) { return; }
    $task = new \plagiarism_yourbrand\task\submit_task();
    $task->set_custom_data([
        'cmid' => $event->contextinstanceid,
        'userid' => $event->userid,
        'itemid' => $event->objectid,
        'pathnamehashes' => $event->other['pathnamehashes'] ?? [],
    ]);
    \core\task\manager::queue_adhoc_task($task, true);   // no HTTP here
}
```

### 7.4 Adhoc task → submit; scheduled task → poll status, then store results
- `submit_task`: for each `pathnamehash`, load the `stored_file`, call
  `api_client::analyze(...)`, save `{cmid,userid,fileid,batch_id,status='submitted'}`.
- `poll_task` (cron every few minutes): for rows in `submitted`, call
  `status(batch_id)`; when `done`, call `results(batch_id)`, take the top
  `plagiarism_analysis[].similarity` → store `score` + `status='scored'`.

### 7.5 `get_links()` — badge in the grading view
Look up the stored row for `{cmid, userid, file pathnamehash}`; if `scored`,
render a coloured percentage badge (green/amber/red by your thresholds) plus a
link to a report page (you can embed the CSV/PDF export or your own view built
from the `results` JSON + `matches` spans). Gate it behind a
`plagiarism/yourbrand:viewreport` capability.

---

## 8. Endpoint quick reference

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/auth/register` | create the service account |
| POST | `/api/v1/auth/jwt/login` | get bearer token (form: username,password) |
| GET  | `/api/v1/users/me` | validate token |
| POST | `/api/v1/assignments` | create assignment (map `external_ref`=cmid) |
| GET  | `/api/v1/assignments` | list assignments |
| POST | `/api/v1/analyze` | submit file(s)/text → `batch_id` |
| GET  | `/api/v1/batches/{id}/status` | poll progress |
| GET  | `/api/v1/batches/{id}/results` | per-document scores + matches |
| GET  | `/api/v1/batches/{id}/report.csv` / `.pdf` | exports |
| GET  | `/api/v1/ai-detection/health` | AI provider health |

---

## 9. Limits, errors, and privacy

- **Files:** `.pdf .docx .txt .png .jpg .jpeg`, ≤ **25 MB** each, ≤ **50 files/batch**.
- **Errors:** `400` bad file type / bad options / bad credentials; `401` missing or
  expired token (re-login); `404` unknown batch or not owned by your account.
- **AI detection is advisory**, never proof — surface the `note` to graders and do
  not auto-penalise on it.
- **Privacy/GDPR:** submission content leaves Moodle for this external service.
  Declare it in your plugin's `privacy` provider (external location) and disclose
  it to students; provide export/delete. Prefer a per-institution service account
  and HTTPS.

---

## 10. Notes for the API owner (optional hardening)

The current auth is user JWT (1 h). For robust machine-to-machine Moodle use,
consider adding: long-lived **API keys / service tokens** (so the plugin doesn't
store a password), a **webhook** ("batch done" callback) to avoid polling, and an
explicit **delete** endpoint for GDPR erasure requests.
```
