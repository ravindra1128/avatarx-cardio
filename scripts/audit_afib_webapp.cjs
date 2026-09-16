/* Offline contract probes against the user's webapp source. No browser,
 * capture, network, or writes to the webapp. BaseScanProcessor is stubbed;
 * these probes exercise only the storage and dispatch methods shown below.
 */
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = process.argv[2];
if (!root) throw new Error("Usage: node audit_afib_webapp.cjs /path/to/webapp");
const read = (name) => fs.readFileSync(path.join(root, "src", name), "utf8");
const source = read("Pages/CardioStaging/CardioStagingScan.jsx");
const numberFn = source.match(/function shenaiNum\(v\) \{[\s\S]*?\n\}/)?.[0];
if (!numberFn) throw new Error("Snapshot number helper changed; inspect the probe");
const stored = new Map();
const context = vm.createContext({
  console: { log() {}, warn() {} }, Date, Promise, setTimeout, clearTimeout,
  STAGING_RESULT_KEY: "afib_result_staging",
  BaseScanProcessor: class { async onScanCancel() {} },
  sessionStorage: {
    setItem(key, value) { stored.set(key, value); },
    getItem(key) { return stored.get(key) ?? null; },
    removeItem(key) { stored.delete(key); },
  },
});
const processor = read("lib/scan/staging/AFibProcessor.js")
  .replace(/^import .*;\s*$/gm, "").replace("export class AFibProcessor", "class AFibProcessor");
const orchestrator = read("lib/scan/ScanOrchestrator.js")
  .replace("export class ScanOrchestrator", "class ScanOrchestrator");
vm.runInContext(`${numberFn}\n${processor}\n${orchestrator}`, context);

(async () => {
  const report = await vm.runInContext(`(async () => {
    const log = () => {};
    const options = {enabled: true, url: "https://offline.invalid", getClip: () => null};
    const scanA = new AFibProcessor(options);
    const scanB = new AFibProcessor(options);
    await scanB.onScanStart({warn: log, log});
    scanB._settle({ok: true, data: {scan_id: "new-B", outcome: "ACCEPT", predicted_class: "SINUS"}});
    const before = JSON.parse(sessionStorage.getItem(STAGING_RESULT_KEY));
    scanA._settle({ok: true, data: {scan_id: "old-A", outcome: "ACCEPT", predicted_class: "AFIB_SUGGESTIVE"}});
    const after = JSON.parse(sessionStorage.getItem(STAGING_RESULT_KEY));
    let finishedCalls = 0;
    const disabled = new ScanOrchestrator({sdk: {}, processors: [{id: "afib", enabled: false,
      onScanStart: async () => {}, onScanFinish: async () => {finishedCalls++;}}]});
    await disabled.start();
    const results = await disabled.finish();
    return {
      notice: "Software counterexamples using stubbed lifecycle dependencies; not a browser or clinical test.",
      explicitNullNumber: shenaiNum(null), missingNumber: shenaiNum(undefined),
      disabledProcessor: {finishedCalls, results},
      latePreviousScan: {before: before.data.scan_id, after: after.data.scan_id},
    };
  })()`, context);
  process.stdout.write(JSON.stringify(report, null, 2) + "\n");
})();
