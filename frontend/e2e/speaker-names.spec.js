import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';

test('demo fixture without attendees displays anonymous speakers and cited summary', async ({ page }) => {
  const fixture = JSON.parse(await readFile('../docs/SPEAKER_NAME_EVIDENCE.json', 'utf8'));
  await page.addInitScript(() => sessionStorage.setItem('access_token', 'offline-ui-test'));
  await page.route('**/api/**', route => {
    const url = new URL(route.request().url());
    const body = url.pathname.endsWith('/config') ? { demo_mode: true } : url.pathname.endsWith('/media-url') ? { url: '/unused-media' } : fixture;
    return route.fulfill({ json: body });
  });
  await page.goto('/meeting.html?id=speaker-audit');
  await expect(page.locator('.transcript strong')).toHaveText(['SPEAKER_01','SPEAKER_11','SPEAKER_01']);
  await expect(page.locator('.summary-editor')).toHaveValue(fixture.summary);
  await expect(page.getByText('No actions extracted.', { exact: true })).toBeVisible();
  await expect(page.locator('.transcript').getByText('inferred', {exact:true})).toHaveCount(0);
  await expect(page.locator('.evidence-chip').first()).toBeVisible();
});

test('inferred speaker and owner labels are visible with owner evidence', async ({ page }) => {
  const fixture = JSON.parse(await readFile('../docs/SPEAKER_NAME_EVIDENCE.json', 'utf8'));
  fixture.transcript[0].speaker_name = 'Ravi';
  fixture.transcript[0].speaker_name_source = 'inferred';
  fixture.intelligence.actionItems = [{ text: 'Send the report', owner: 'Ravi', owner_name_source: 'inferred', source_segment_ids: [fixture.transcript[1].segment_id] }];
  await page.addInitScript(() => sessionStorage.setItem('access_token', 'offline-ui-test'));
  await page.route('**/api/**', route => route.fulfill({json: route.request().url().endsWith('/config') ? {demo_mode:true} : route.request().url().endsWith('/media-url') ? {url:'/unused-media'} : fixture}));
  await page.goto('/meeting.html?id=speaker-audit');
  await expect(page.locator('.transcript').getByText('inferred',{exact:true})).toBeVisible();
  await expect(page.getByText('Send the report — Ravi (inferred)')).toBeVisible();
  const action = page.locator('.block').filter({has:page.getByRole('heading',{name:'Actions',exact:true})});
  await expect(action.getByRole('button',{name:`Play evidence ${fixture.transcript[1].segment_id}`})).toBeVisible();
});
