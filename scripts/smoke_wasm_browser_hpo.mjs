#!/usr/bin/env node
// Actual Chrome Web Workers, each loading an independent web-target WASM instance.
import {createServer} from "node:http";
import {readFile, mkdtemp, rm} from "node:fs/promises";
import {spawn} from "node:child_process";
import {fileURLToPath} from "node:url";
import path from "node:path";
import os from "node:os";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const pkg = path.resolve(process.argv[2] || path.join(repo, "crates/dag-ml-wasm/pkg-web"));
const files = new Map([
  ["/", [path.join(repo, "scripts/browser_hpo/index.html"), "text/html"]],
  ["/worker.js", [path.join(repo, "scripts/browser_hpo/worker.js"), "text/javascript"]],
  ["/ridge_operator.js", [path.join(repo, "scripts/browser_hpo/ridge_operator.js"), "text/javascript"]],
  ["/ridge_refit_oracle.js", [path.join(repo, "scripts/wasm_ridge_refit_oracle.mjs"), "text/javascript"]],
  ["/initial_refit_fixture.json", [path.join(repo, "crates/dag-ml-core/tests/fixtures/initial_full_refit/package.json"), "application/json"]],
  ["/fixture.json", [path.join(repo, "crates/dag-ml-core/tests/fixtures/package/data/coordinator_data_plan_envelope_sample12.json"), "application/json"]],
  ["/pkg/dag_ml_wasm.js", [path.join(pkg, "dag_ml_wasm.js"), "text/javascript"]],
  ["/pkg/dag_ml_wasm_bg.wasm", [path.join(pkg, "dag_ml_wasm_bg.wasm"), "application/wasm"]],
]);
const server = createServer(async (request, response) => {
  const item = files.get(new URL(request.url, "http://localhost").pathname);
  if (!item) { response.writeHead(404).end(); return; }
  try {
    const bytes = await readFile(item[0]);
    response.writeHead(200, {"Content-Type": item[1], "Cache-Control": "no-store"}).end(bytes);
  } catch (error) {
    response.writeHead(500).end(String(error));
  }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));

try {
  const url = `http://127.0.0.1:${server.address().port}/`;
  const chrome = process.env.CHROME_BIN || "google-chrome";
  const profile = await mkdtemp(path.join(os.tmpdir(), "dagml-browser-hpo-"));
  const args = ["--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
    "--disable-background-networking", "--remote-debugging-port=0",
    "--remote-allow-origins=*", `--user-data-dir=${profile}`, url];
  const child = spawn(chrome, args, {stdio: ["ignore", "ignore", "pipe"]});
  let stderr = "";
  child.stderr.on("data", chunk => { stderr += chunk; });
  try {
    const deadline = Date.now() + 45000;
    let debuggerUrl;
    while (!debuggerUrl && Date.now() < deadline) {
      debuggerUrl = stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/)?.[1];
      if (!debuggerUrl) await new Promise(resolve => setTimeout(resolve, 100));
    }
    if (!debuggerUrl) throw new Error(`Chrome DevTools did not start: ${stderr.slice(-1200)}`);
    const debuggerPort = new URL(debuggerUrl.replace("ws:", "http:")).port;
    let page;
    while (!page && Date.now() < deadline) {
      const targets = await (await fetch(`http://127.0.0.1:${debuggerPort}/json/list`)).json();
      page = targets.find(target => target.type === "page" && target.url === url);
      if (!page) await new Promise(resolve => setTimeout(resolve, 100));
    }
    if (!page) throw new Error("Chrome did not open the HPO smoke page");
    const websocket = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      websocket.addEventListener("open", resolve, {once: true});
      websocket.addEventListener("error", reject, {once: true});
    });
    try {
      let status = "pending";
      let details = "";
      for (let id = 1; Date.now() < deadline; id++) {
        const reply = new Promise(resolve => {
          const onMessage = event => {
            const message = JSON.parse(event.data);
            if (message.id !== id) return;
            websocket.removeEventListener("message", onMessage);
            resolve(message);
          };
          websocket.addEventListener("message", onMessage);
        });
        websocket.send(JSON.stringify({id, method: "Runtime.evaluate", params: {
          expression: "document.body.dataset.result + '|' + document.body.textContent",
          returnByValue: true,
        }}));
        const value = (await reply).result?.result?.value;
        if (typeof value === "string") {
          [status, details] = [value.split("|", 1)[0], value.slice(value.indexOf("|") + 1)];
        }
        if (status === "pass" || status === "fail") break;
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      if (status !== "pass") throw new Error(`Chrome Web Worker HPO smoke ${status}: ${details.slice(0, 1800)}\n${stderr.slice(-700)}`);
    } finally {
      websocket.close();
    }
  } finally {
    child.kill();
    await rm(profile, {recursive: true, force: true, maxRetries: 3});
  }
  process.stdout.write("Chrome ridge HPO parallel/pruning/resume, REFIT and replay: PASS\n");
} finally {
  await new Promise(resolve => server.close(resolve));
}
