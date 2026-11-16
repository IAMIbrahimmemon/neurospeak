// Needs playwright-core and Chrome: npm i playwright-core
import { chromium } from "playwright-core";
import { fileURLToPath } from "node:url";
const b = await chromium.launch({ channel: "chrome" });
const page = await b.newPage({ viewport: { width: 1200, height: 630 } });
await page.goto("file://" + fileURLToPath(new URL("card.html", import.meta.url)));
await page.evaluate(() => document.fonts.ready);
await page.screenshot({ path: process.argv[2] });
await b.close();
