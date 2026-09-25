// Modified by RV-SDTM contributors for this public release; see NOTICE.
// Browser smoke check for the published gallery; never loads a model or dataset.
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const url = process.argv[2];
if (!url || !/^https?:\/\//.test(url)) {
  throw new Error('Usage: node tools/check_gallery_playback.cjs <gallery URL>');
}
const scenes = ['full_bev', '01_cyclists', '02_pedestrians', '03_far_vru', '04_vehicles'];

async function openGallery(page, target) {
  for (let attempt = 0; attempt < 6; attempt++) {
    const response = await page.goto(target, { waitUntil: 'domcontentloaded', timeout: 45000 });
    if (response && response.ok()) return;
    if (attempt === 5) throw new Error(`Gallery HTTP status: ${response && response.status()}`);
    await page.waitForTimeout(5000);
  }
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const desktop = await browser.newContext({
      viewport: { width: 1280, height: 900 }, reducedMotion: 'reduce',
    });
    const page = await desktop.newPage();
    await openGallery(page, url);
    assert.equal(await page.locator('video').count(), 5);
    for (const id of scenes) {
      const video = page.locator(`article[id="${id}"] video`);
      await video.scrollIntoViewIfNeeded();
      await video.evaluate(element => {
        element.play().catch(error => { element.dataset.playError = error.message; });
      });
      await page.waitForFunction(scene => {
        const video = document.getElementById(scene).querySelector('video');
        if (video.error || video.dataset.playError) {
          throw new Error(video.dataset.playError || video.error.message);
        }
        return video.readyState >= 2 && video.currentTime > 0.25;
      }, id, { timeout: 45000 });
      const state = await video.evaluate(element => {
        element.pause();
        return { width: element.videoWidth, height: element.videoHeight,
          duration: element.duration, time: element.currentTime, controls: element.controls };
      });
      assert.equal(state.width, 1920);
      assert.equal(state.height, 1080);
      assert.equal(state.controls, true);
      assert.ok(Math.abs(state.duration - (id === 'full_bev' ? 12 : 13)) < 0.1);
      console.log(`[PASS] ${id}: decoded and played 1080p video, ${state.duration}s`);
    }
    await desktop.close();

    const mobile = await browser.newContext({
      viewport: { width: 390, height: 844 }, reducedMotion: 'no-preference',
    });
    const phone = await mobile.newPage();
    await openGallery(phone, `${url.replace(/#.*$/, '')}#01_cyclists`);
    await phone.locator('article[id="01_cyclists"]').scrollIntoViewIfNeeded();
    await phone.waitForFunction(() => {
      const video = document.getElementById('01_cyclists').querySelector('video');
      return !video.paused && video.currentTime > 0.25;
    }, null, { timeout: 45000 });
    assert.ok(await phone.evaluate(() =>
      document.documentElement.scrollWidth <= window.innerWidth + 1));
    console.log('[PASS] Mobile layout has no horizontal overflow; visible muted video autoplays');
    await mobile.close();
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
