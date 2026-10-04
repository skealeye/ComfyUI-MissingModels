// Missing models: download a workflow's missing models into the right ComfyUI/models folders.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TITLE = "Missing models";
const fmt = (b) => (b == null ? "?" : b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(1)} MB`);
let dlg = null;

function el(tag, props = {}, ...kids) {
  const e = Object.assign(document.createElement(tag), props);
  for (const k of kids) e.append(k);
  return e;
}

function toast(detail, severity = "info") {
  const t = app.extensionManager?.toast;
  if (t) t.add({ severity, summary: TITLE, detail, life: 8000 });
  else console.log(`[${TITLE}]`, detail);
}

async function post(path, body) {
  const r = await api.fetchApi(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                       body: JSON.stringify(body) });
  return { status: r.status, data: await r.json() };
}

async function check() {
  const workflow = (app.rootGraph ?? app.graph).serialize();
  return (await post("/missing_models/check", { workflow })).data.missing ?? [];
}

function close() {
  dlg?.remove();
  dlg = null;
}

function show(rows) {
  close();
  const status = el("div", { style: "margin-top:8px;min-height:1.4em" });
  const bar = el("progress", { max: 1, value: 0, style: "width:100%;display:none" });
  const boxes = [];
  const table = el("table", { style: "width:100%;border-collapse:collapse;font-size:13px" });
  table.append(el("tr", {}, ...["", "File", "Folder", "Size", ""].map((h) => el("th", { textContent: h,
    style: "text-align:left;padding:4px 6px;border-bottom:1px solid var(--border-color)" }))));
  for (const r of rows) {
    const box = el("input", { type: "checkbox", checked: r.downloadable, disabled: !r.downloadable });
    boxes.push([box, r]);
    const stem = r.name.split("/").pop().replace(/\.[^.]+$/, "");
    const note = r.downloadable ? el("span", { textContent: new URL(r.url).host, style: "opacity:.7" })
      : el("span", {}, `${r.reason}  `,
          el("a", { href: `https://huggingface.co/search/full-text?q=${encodeURIComponent(r.name)}`, target: "_blank",
                    textContent: "Search Hugging Face" }), "  ",
          el("a", { href: `https://civitai.com/search/models?query=${encodeURIComponent(stem)}`, target: "_blank",
                    textContent: "Search Civitai" }));
    const row = el("tr", { style: r.downloadable ? "" : "opacity:.6" },
      el("td", {}, box), el("td", { textContent: r.name }), el("td", { textContent: r.directory }),
      el("td", { textContent: fmt(r.size) }), el("td", {}, note));
    for (const td of row.children) td.style.padding = "4px 6px";
    table.append(row);
  }
  const go = el("button", { textContent: "Download", disabled: !rows.some((r) => r.downloadable) });
  const stop = el("button", { textContent: "Cancel", disabled: true });
  const shut = el("button", { textContent: "Close" });
  shut.onclick = close;
  stop.onclick = () => post("/missing_models/cancel", {});
  go.onclick = async () => {
    const items = boxes.filter(([b]) => b.checked).map(([, r]) => ({ name: r.name, directory: r.directory, url: r.url }));
    if (!items.length) return;
    const res = await post("/missing_models/download", { items });
    if (res.status !== 200) { status.textContent = res.data.error; return; }
    go.disabled = true; stop.disabled = false; bar.style.display = "";
    for (const [b] of boxes) b.disabled = true;
  };
  dlg = el("div", { style: "position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:10000;display:flex;" +
                           "align-items:center;justify-content:center" },
    el("div", { style: "background:var(--comfy-menu-bg);color:var(--fg-color);border:1px solid var(--border-color);" +
                       "border-radius:8px;padding:16px;min-width:640px;max-width:90vw;max-height:80vh;overflow:auto" },
      el("h3", { textContent: rows.length ? `${TITLE} (${rows.length})` : `${TITLE}: none`, style: "margin:0 0 8px" }),
      rows.length ? table : el("div", { textContent: "Everything this workflow needs is installed." }),
      bar, status, el("div", { style: "margin-top:12px;display:flex;gap:8px;justify-content:flex-end" }, go, stop, shut)));
  dlg.ui = { status, bar, go, stop };
  document.body.append(dlg);
}

async function open() {
  try { show(await check()); } catch (e) { toast(`Check failed: ${e}`, "error"); }
}

api.addEventListener("missing_models.progress", ({ detail: d }) => {
  if (!dlg) return;
  dlg.ui.bar.max = d.total || 1; dlg.ui.bar.value = d.done;
  dlg.ui.status.textContent = `${d.index + 1} of ${d.count}: ${d.name}  ${fmt(d.done)} / ${fmt(d.total)}`;
});

api.addEventListener("missing_models.done", async ({ detail }) => {
  const res = detail.results ?? [];
  const ok = res.filter((r) => r.status === "installed").length;
  await app.refreshComboInNodes?.();
  toast(`${ok} of ${res.length} installed`, ok === res.length ? "success" : "warn");
  if (!dlg) return;
  dlg.ui.stop.disabled = true; dlg.ui.bar.style.display = "none";
  dlg.ui.status.replaceChildren(...res.map((r) => el("div", {
    textContent: `${r.status === "installed" ? "✓" : "✗"} ${r.name}: ${r.status === "installed" ? r.path : r.error ?? r.status}`,
  })));
});

app.registerExtension({
  name: "MissingModels",
  commands: [{ id: "MissingModels.Open", label: TITLE, icon: "pi pi-download", function: open }],
  menuCommands: [{ path: ["Extensions"], commands: ["MissingModels.Open"] }],
  actionBarButtons: [{ icon: "pi pi-download", label: TITLE, tooltip: "Download this workflow's missing models",
                       onClick: open }],
  async afterConfigureGraph() {
    try {
      const rows = await check();
      const n = rows.filter((r) => r.downloadable).length;
      if (rows.length) toast(`${rows.length} missing (${n} downloadable): use the ${TITLE} button`, "warn");
    } catch (e) { console.warn(`[${TITLE}]`, e); }
  },
});
