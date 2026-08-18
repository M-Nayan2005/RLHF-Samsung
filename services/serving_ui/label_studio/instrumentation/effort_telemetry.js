/*
 * effort_telemetry.js — Label Studio Frontend instrumentation for Tier 2.
 *
 * Captures the three E-DRDE effort terms while a junior annotator corrects a
 * wiggled mask, and ships them to serving_ui just before the annotation submits:
 *
 *   click_count           C        vertex placements and boundary clicks
 *   cursor_path_length_px L_path   summed Euclidean cursor travel
 *   dwell_time_ms         T_dwell  active pointer time over the mask boundary
 *
 * Label Studio's stock ANNOTATION_UPDATED webhook carries none of this — it has
 * the final polygon and `lead_time`, and nothing about how the human got there.
 * The whole system rests on correction effort being the reward signal, so if
 * this file does not run, the reward is not merely degraded, it is absent.
 *
 * ---------------------------------------------------------------------------
 * How it hooks in
 * ---------------------------------------------------------------------------
 * Deliberately NOT bound to Label Studio Frontend internals. LSF's MobX store
 * shape (`window.Htx`, `annotationStore`, region classes) changes between minor
 * versions and is not a public API; instrumentation written against it breaks on
 * upgrade, silently, in a way that looks like "annotators produced no telemetry
 * today". Instead this hooks two things that are stable:
 *
 *   1. DOM pointer events on the image canvas  — for C, L_path, T_dwell.
 *   2. `fetch` / `XMLHttpRequest` to the annotation-create endpoint — for
 *      knowing exactly when a submit is happening.
 *
 * Hooking the network layer also means the beacon fires on EVERY submit path:
 * the Submit button, the Update button, and the keyboard shortcut, without
 * having to find and bind three different DOM nodes.
 *
 * ---------------------------------------------------------------------------
 * Transport
 * ---------------------------------------------------------------------------
 * PRIMARY — `beacon`: POST to serving_ui `/telemetry/raw`, which enriches the
 * payload and forwards it to Dev 4's gateway. Chosen because stock Label Studio
 * exposes no supported hook for writing `annotation.meta` before submit, and
 * because the frozen `LSAnnotationUpdatedPayload` declares `effort_telemetry` at
 * the TOP LEVEL — so the beacon matches the contract Dev 4 validates against,
 * while meta injection would require Dev 4 to lift a nested block first
 * (divergence D5). See services/serving_ui/README.md.
 *
 * SECONDARY — `ls_meta`: when enabled, the annotation POST body is rewritten in
 * flight to carry `meta.effort_telemetry`. Whether that survives depends on the
 * Label Studio version having a writable `meta` field on its Annotation model,
 * which is not guaranteed. Never rely on it alone.
 *
 * ---------------------------------------------------------------------------
 * Configuration
 * ---------------------------------------------------------------------------
 * Set `window.RLHF_TELEMETRY_CONFIG` before this script loads, or put data-*
 * attributes on the <script> tag. See infra/nginx/inject.conf.
 *
 *   endpoint   telemetry receiver           default "http://localhost:8003/telemetry/raw"
 *   transport  "beacon" | "ls_meta" | "both"           default "beacon"
 *   boundaryPx dwell proximity threshold, CSS px       default 24
 *   debug      log to console                          default false
 */
(function () {
  "use strict";

  // ------------------------------------------------------------------ config

  var script = document.currentScript;
  var attr = function (name, fallback) {
    if (script && script.dataset && script.dataset[name] != null && script.dataset[name] !== "") {
      return script.dataset[name];
    }
    return fallback;
  };

  var USER = window.RLHF_TELEMETRY_CONFIG || {};
  var CFG = {
    endpoint: USER.endpoint || attr("endpoint", "http://localhost:8003/telemetry/raw"),
    transport: USER.transport || attr("transport", "beacon"),
    boundaryPx: parseFloat(USER.boundaryPx || attr("boundaryPx", "24")),
    debug: String(USER.debug != null ? USER.debug : attr("debug", "false")) === "true"
  };

  var log = function () {
    if (!CFG.debug) return;
    var args = Array.prototype.slice.call(arguments);
    console.log.apply(console, ["[rlhf-telemetry]"].concat(args));
  };

  if (window.__RLHF_TELEMETRY_INSTALLED__) {
    log("already installed, skipping");
    return;
  }
  window.__RLHF_TELEMETRY_INSTALLED__ = true;

  // ------------------------------------------------------------------- state

  var SESSION_ID = "sess_" + Math.random().toString(36).slice(2, 12);

  /* One counter set per Label Studio task. Keyed by LS numeric task id, because
   * an annotator can move between tasks without a page reload in the label
   * stream — resetting on navigation instead of keying would merge two people's
   * worth of effort into whichever task submitted last. */
  var counters = Object.create(null);

  function countersFor(lsTaskId) {
    var key = String(lsTaskId == null ? "unknown" : lsTaskId);
    if (!counters[key]) {
      counters[key] = {
        clickCount: 0,
        pathLengthCss: 0,
        pathLengthImage: 0,
        dwellMs: 0,
        openedAt: Date.now(),
        sent: false,
        sentWithId: false
      };
    }
    return counters[key];
  }

  var pointer = { x: null, y: null, ts: null, nearBoundary: false };

  /* Task context, refreshed whenever the visible task changes. */
  var ctx = {
    lsTaskId: null,
    taskId: null,
    projectId: null,
    wiggleSeed: null,
    polygonPercent: null
  };

  // -------------------------------------------------------------- LS task API

  function currentLsTaskId() {
    /* Label Studio puts the task id in the query string in the data manager
     * (?task=123) and in the path in the label stream (/projects/1/data?task=123
     * or .../quickview?task=123). Fall back to whatever the last annotation POST
     * told us. */
    var match = /[?&]task=(\d+)/.exec(window.location.search || "");
    if (match) return parseInt(match[1], 10);
    match = /\/tasks\/(\d+)/.exec(window.location.pathname || "");
    if (match) return parseInt(match[1], 10);
    return ctx.lsTaskId;
  }

  function refreshContext(lsTaskId) {
    if (lsTaskId == null || lsTaskId === ctx.lsTaskId) return Promise.resolve(ctx);

    return fetch("/api/tasks/" + lsTaskId + "/", { credentials: "same-origin" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (task) {
        if (!task) return ctx;

        ctx.lsTaskId = lsTaskId;
        ctx.projectId = task.project != null ? String(task.project) : ctx.projectId;
        ctx.taskId = (task.data && task.data.task_id) || null;

        var prediction = pickPrediction(task.predictions);
        ctx.wiggleSeed = prediction ? seedFromPrediction(prediction) : null;
        ctx.polygonPercent = prediction ? polygonFromPrediction(prediction) : null;

        log("context", ctx);
        if (!ctx.taskId) {
          console.warn(
            "[rlhf-telemetry] LS task " + lsTaskId + " has no data.task_id. Telemetry " +
            "cannot be tied back to a QueueTask; check how tasks were imported."
          );
        }
        return ctx;
      })
      .catch(function (err) {
        log("context refresh failed", err);
        return ctx;
      });
  }

  function pickPrediction(predictions) {
    if (!predictions || !predictions.length) return null;
    /* Newest wins: a task re-served after a consensus requeue carries an older
     * prediction whose seed describes an action this annotator never saw. */
    return predictions.slice().sort(function (a, b) {
      return new Date(b.created_at || 0) - new Date(a.created_at || 0);
    })[0];
  }

  function seedFromPrediction(prediction) {
    /* Channel 2: model_version, "serving-ui-stochastic-0.1.0|seed=<hex>". */
    var mv = prediction.model_version || "";
    var idx = mv.indexOf("|seed=");
    if (idx !== -1) return mv.slice(idx + 6) || null;

    /* Channel 1: region meta.text, "wiggle_seed=<hex>". */
    var regions = prediction.result || [];
    for (var i = 0; i < regions.length; i++) {
      var texts = (regions[i].meta && regions[i].meta.text) || [];
      for (var j = 0; j < texts.length; j++) {
        var m = /wiggle_seed=([0-9a-zA-Z_\-]+)/.exec(String(texts[j]));
        if (m) return m[1];
      }
    }
    /* Channel 3 (serving_ui /served/{task_id}) is the server's job — it fills a
     * missing seed in during enrichment, so there is nothing to do here. */
    return null;
  }

  function polygonFromPrediction(prediction) {
    var regions = prediction.result || [];
    for (var i = 0; i < regions.length; i++) {
      var value = regions[i].value || {};
      if (value.points && value.points.length >= 3) return value.points;
    }
    return null;
  }

  // ------------------------------------------------------- geometry (screen)

  function imageElement() {
    /* The rendered image inside the LSF canvas. Label Studio has used several
     * class names for it, so match on the element rather than on styling. */
    var candidates = document.querySelectorAll(
      "img[alt='LS'], .lsf-image__image, .lsf-image img, img.image-element, canvas + img, img"
    );
    for (var i = 0; i < candidates.length; i++) {
      var el = candidates[i];
      var rect = el.getBoundingClientRect();
      if (rect.width > 80 && rect.height > 80) return el;
    }
    return null;
  }

  function imageScale(el) {
    /* CSS pixels per image pixel. naturalWidth is the true image width, which is
     * what the polygon percentages are relative to. */
    if (!el || !el.naturalWidth) return null;
    var rect = el.getBoundingClientRect();
    return {
      rect: rect,
      sx: rect.width / el.naturalWidth,
      sy: rect.height / el.naturalHeight,
      naturalWidth: el.naturalWidth,
      naturalHeight: el.naturalHeight
    };
  }

  function distanceToBoundaryCss(clientX, clientY) {
    /* Shortest distance from the pointer to the served polygon's outline, in CSS
     * pixels. Returns null when the polygon or the image cannot be located,
     * which suppresses dwell accumulation rather than guessing. */
    if (!ctx.polygonPercent) return null;
    var el = imageElement();
    var scale = imageScale(el);
    if (!scale) return null;

    var pts = ctx.polygonPercent;
    var best = Infinity;

    for (var i = 0; i < pts.length; i++) {
      var a = pts[i];
      var b = pts[(i + 1) % pts.length];
      var ax = scale.rect.left + (a[0] / 100) * scale.rect.width;
      var ay = scale.rect.top + (a[1] / 100) * scale.rect.height;
      var bx = scale.rect.left + (b[0] / 100) * scale.rect.width;
      var by = scale.rect.top + (b[1] / 100) * scale.rect.height;
      best = Math.min(best, pointSegmentDistance(clientX, clientY, ax, ay, bx, by));
    }
    return best;
  }

  function pointSegmentDistance(px, py, ax, ay, bx, by) {
    var dx = bx - ax;
    var dy = by - ay;
    var lenSq = dx * dx + dy * dy;
    if (lenSq < 1e-9) return Math.hypot(px - ax, py - ay);
    var t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  // ------------------------------------------------------------- pointer track

  /* Dwell is capped per pointermove so that a pointer parked over the boundary
   * while the annotator reads the instructions, takes a call, or leaves for
   * lunch does not register as hours of effort. T_dwell is meant to measure
   * hesitation over a difficult edge, not wall-clock time with the tab open. */
  var MAX_DWELL_STEP_MS = 500;

  function onPointerMove(event) {
    var now = performance.now();
    var counter = countersFor(currentLsTaskId());

    if (pointer.x != null) {
      var dxCss = event.clientX - pointer.x;
      var dyCss = event.clientY - pointer.y;
      var stepCss = Math.hypot(dxCss, dyCss);

      /* Ignore teleports: a pointer jump larger than this is a window switch or
       * a scroll-induced coordinate shift, not cursor travel. */
      if (stepCss < 400) {
        counter.pathLengthCss += stepCss;

        var scale = imageScale(imageElement());
        if (scale && scale.sx > 0 && scale.sy > 0) {
          counter.pathLengthImage += Math.hypot(dxCss / scale.sx, dyCss / scale.sy);
        }
      }

      var distance = distanceToBoundaryCss(event.clientX, event.clientY);
      var near = distance != null && distance <= CFG.boundaryPx;
      if (near && pointer.nearBoundary && pointer.ts != null) {
        counter.dwellMs += Math.min(now - pointer.ts, MAX_DWELL_STEP_MS);
      }
      pointer.nearBoundary = near;
    }

    pointer.x = event.clientX;
    pointer.y = event.clientY;
    pointer.ts = now;
  }

  function onPointerDown(event) {
    /* Count clicks landing on the labeling surface only. A click on the sidebar,
     * the label picker, or the submit button is navigation, not correction
     * effort, and C is supposed to measure the latter. */
    if (!event.target || !event.target.closest) return;
    var onCanvas = event.target.closest(
      ".lsf-main-view, .lsf-canvas, .lsf-image, .ls-main-view, [class*='ImageView']"
    );
    if (!onCanvas) return;
    countersFor(currentLsTaskId()).clickCount += 1;
  }

  function onBlur() {
    /* Stop accumulating dwell when the tab loses focus. */
    pointer.nearBoundary = false;
    pointer.ts = null;
  }

  document.addEventListener("pointermove", onPointerMove, { passive: true, capture: true });
  document.addEventListener("pointerdown", onPointerDown, { passive: true, capture: true });
  window.addEventListener("blur", onBlur);
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) onBlur();
  });

  // --------------------------------------------------------------- payload

  function buildPayload(lsTaskId, annotationId) {
    var counter = countersFor(lsTaskId);
    return {
      annotation_id: annotationId != null ? String(annotationId) : null,
      task_id: ctx.taskId || ("ls_task_" + lsTaskId),
      effort_telemetry: {
        click_count: Math.round(counter.clickCount),
        cursor_path_length_px: Math.round(counter.pathLengthCss * 10) / 10,
        dwell_time_ms: Math.round(counter.dwellMs),
        wiggle_seed: ctx.wiggleSeed || null
      },
      cursor_path_length_image_px: Math.round(counter.pathLengthImage * 10) / 10,
      project_id: ctx.projectId,
      ls_task_id: lsTaskId != null ? Number(lsTaskId) : null,
      client_session_id: SESSION_ID,
      wiggle_seed: ctx.wiggleSeed || null,
      lead_time: Math.round((Date.now() - counter.openedAt) / 100) / 10,
      client_sent_at: new Date().toISOString(),
      transport: CFG.transport === "ls_meta" ? "ls_meta" : "beacon"
    };
  }

  function send(payload, lsTaskId) {
    var counter = countersFor(lsTaskId);

    /* Two senders race on every submit: the teardown flush (no annotation_id)
     * and the submit response (has it). On "submit and next" Label Studio
     * navigates as soon as the request is away, so the teardown flush usually
     * wins - which is why a first beacon normally carries a null annotation_id.
     *
     * Rather than suppress one of them, a beacon that arrives WITH an id is
     * allowed through as an upgrade, exactly once. The downstream join is on
     * wiggle_seed, which is identical on both, so the second record replaces the
     * first rather than duplicating it. Sending twice is cheap; losing the id is
     * not, because it is the only exact key back to the annotation. */
    var hasId = payload.annotation_id != null;
    if (counter.sent && !(hasId && !counter.sentWithId)) {
      log("already sent for task", lsTaskId, hasId ? "(id already sent)" : "(no new id)");
      return;
    }
    if (counter.sent && hasId) {
      log("upgrading earlier beacon with annotation_id", payload.annotation_id);
      payload.supersedes_prior_beacon = true;
    }
    counter.sent = true;
    counter.sentWithId = counter.sentWithId || hasId;

    var body = JSON.stringify(payload);
    log("sending", payload);

    /* sendBeacon survives the page teardown that follows a submit-and-advance.
     * A plain fetch can be cancelled mid-flight when LSF navigates to the next
     * task, which loses the telemetry for exactly the tasks that were completed
     * fastest — a silent, systematic bias in the reward signal. */
    var sent = false;
    if (navigator.sendBeacon) {
      try {
        /* text/plain, not application/json. It is the only JSON-carrying content
         * type on the CORS-safelist, so the beacon goes straight out instead of
         * needing a preflight — and a preflight that fails makes sendBeacon fail
         * *silently*. serving_ui reads the body raw for exactly this reason. */
        sent = navigator.sendBeacon(
          CFG.endpoint,
          new Blob([body], { type: "text/plain;charset=UTF-8" })
        );
      } catch (err) {
        log("sendBeacon threw", err);
      }
    }
    if (!sent) {
      fetch(CFG.endpoint, {
        method: "POST",
        headers: { "Content-Type": "text/plain;charset=UTF-8" },
        body: body,
        keepalive: true,
        mode: "cors"
      }).catch(function (err) {
        console.error("[rlhf-telemetry] telemetry POST failed", err);
      });
    }
  }

  // ---------------------------------------------------- submit interception

  var ANNOTATION_POST = /\/api\/tasks\/(\d+)\/annotations/;
  var ANNOTATION_PATCH = /\/api\/annotations\/(\d+)/;

  function isAnnotationWrite(url, method) {
    var verb = String(method || "GET").toUpperCase();
    if (verb === "GET" || verb === "HEAD") return false;
    return ANNOTATION_POST.test(url) || ANNOTATION_PATCH.test(url);
  }

  function taskIdFromUrl(url) {
    var match = ANNOTATION_POST.exec(url || "");
    return match ? parseInt(match[1], 10) : currentLsTaskId();
  }

  /* Safety net for the case where the page navigates away before the submit
   * response arrives. Sends once, with whatever is known at that moment. */
  var teardownArmed = false;
  function armTeardownFlush(lsTaskId) {
    if (teardownArmed) return;
    teardownArmed = true;
    var flush = function () { send(buildPayload(lsTaskId), lsTaskId); };
    window.addEventListener("pagehide", flush, { once: true });
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) flush();
    }, { once: true });
  }

  function injectMeta(bodyText, payload) {
    /* Rewrite the outgoing annotation body to carry the telemetry as
     * `meta.effort_telemetry`. Best-effort: whether Label Studio persists an
     * arbitrary `meta` object on an annotation, and whether it appears in the
     * webhook, depends on the version's Annotation model. Never the only
     * transport — see the header comment. */
    if (!bodyText) return bodyText;
    try {
      var parsed = JSON.parse(bodyText);
      parsed.meta = parsed.meta || {};
      parsed.meta.effort_telemetry = payload.effort_telemetry;
      return JSON.stringify(parsed);
    } catch (err) {
      log("meta injection skipped, body is not JSON", err);
      return bodyText;
    }
  }

  /* Reading annotation_id off the RESPONSE.
   *
   * Label Studio does not mint an annotation_id until the submit round-trip
   * completes, so a payload sent before the request cannot carry one. But the
   * response to that request *is* the created annotation, id included — so
   * waiting for it lets the telemetry carry the real id and be joined to the
   * webhook exactly, instead of downstream having to match on wiggle_seed or
   * invent a surrogate.
   *
   * The pre-submit path stays as a safety net: if the page tears down before
   * the response lands, `flushOnTeardown` fires whatever was accumulated with a
   * null annotation_id. Losing the id is recoverable (wiggle_seed still joins);
   * losing the whole record is not. */
  var originalFetch = window.fetch;
  window.fetch = function (input, init) {
    var url = "", method = "GET", submitting = false, lsTaskId = null;
    try {
      url = typeof input === "string" ? input : (input && input.url) || "";
      method = (init && init.method) || (input && input.method) || "GET";
      submitting = isAnnotationWrite(url, method);
      if (submitting) {
        lsTaskId = taskIdFromUrl(url);
        armTeardownFlush(lsTaskId);
        if (init && typeof init.body === "string" &&
            (CFG.transport === "ls_meta" || CFG.transport === "both")) {
          init.body = injectMeta(init.body, buildPayload(lsTaskId));
        }
      }
    } catch (err) {
      console.error("[rlhf-telemetry] fetch hook error", err);
    }

    var promise = originalFetch.apply(this, arguments);
    if (!submitting) return promise;

    return promise.then(function (response) {
      try {
        response.clone().json().then(function (body) {
          send(buildPayload(lsTaskId, body && body.id), lsTaskId);
        }).catch(function () {
          send(buildPayload(lsTaskId), lsTaskId);
        });
      } catch (err) {
        send(buildPayload(lsTaskId), lsTaskId);
      }
      return response;
    }, function (err) {
      /* The submit itself failed. Send anyway: the annotator did the work, and
       * an orphaned effort record is more useful than none. */
      send(buildPayload(lsTaskId), lsTaskId);
      throw err;
    });
  };

  var originalOpen = XMLHttpRequest.prototype.open;
  var originalSend = XMLHttpRequest.prototype.send;

  XMLHttpRequest.prototype.open = function (method, url) {
    this.__rlhf = { method: method, url: url };
    return originalOpen.apply(this, arguments);
  };

  XMLHttpRequest.prototype.send = function (body) {
    var self = this;
    try {
      var meta = this.__rlhf;
      if (meta && isAnnotationWrite(meta.url, meta.method)) {
        var lsTaskId = taskIdFromUrl(meta.url);
        armTeardownFlush(lsTaskId);

        this.addEventListener("load", function () {
          var annotationId = null;
          try {
            annotationId = (JSON.parse(self.responseText) || {}).id;
          } catch (err) { /* not JSON; wiggle_seed still joins it */ }
          send(buildPayload(lsTaskId, annotationId), lsTaskId);
        });
        this.addEventListener("error", function () {
          send(buildPayload(lsTaskId), lsTaskId);
        });

        if (typeof body === "string" &&
            (CFG.transport === "ls_meta" || CFG.transport === "both")) {
          return originalSend.call(this, injectMeta(body, buildPayload(lsTaskId)));
        }
      }
    } catch (err) {
      console.error("[rlhf-telemetry] xhr hook error", err);
    }
    return originalSend.apply(this, arguments);
  };

  // ------------------------------------------------------------------ boot

  function poll() {
    var lsTaskId = currentLsTaskId();
    if (lsTaskId != null && lsTaskId !== ctx.lsTaskId) {
      refreshContext(lsTaskId);
    }
  }

  setInterval(poll, 750);
  poll();

  log("installed", CFG);
})();
