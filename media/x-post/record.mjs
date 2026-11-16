// Renders anim.html frame by frame at a fixed clock. Usage: node record.mjs <framesDir> [fps]
import { chromium } from "playwright-core";
import { fileURLToPath } from "node:url";
const dir = process.argv[2], fps = Number(process.argv[3] ?? 30);
const page = await (await chromium.launch({ channel: "chrome" })).newPage({ viewport: { width: 1920, height: 1080 } });
await page.goto("file://" + fileURLToPath(new URL("anim.html", import.meta.url)) + "?t=0");
await page.evaluate(() => document.fonts.ready);
const dur = await page.evaluate(() => window.DURATION);
const n = Math.round(dur * fps);
for (let i = 0; i < n; i++) {
  await page.evaluate((t) => window.render(t), i / fps);
  await page.screenshot({ path: `${dir}/f${String(i).padStart(4, "0")}.png` });
}
await page.evaluate(() => window.render(9.6));
await page.screenshot({ path: `${dir}/../still.png` });
console.log("frames", n);
process.exit(0);
