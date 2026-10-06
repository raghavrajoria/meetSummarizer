import { test, expect } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
const banner='Pre-computed results from earlier pipeline runs. Live hosted-model processing is not connected yet.';
test('real precomputed meetings play, seek, expose tabs and honest provenance',async({page,request})=>{
 const login=await request.post('http://127.0.0.1:8000/auth/login',{data:{username:'demo',password:'demo-password'}});
 const {access_token}=await login.json();
 await page.addInitScript(t=>sessionStorage.setItem('access_token',t),access_token);
 const evidence=[];
 for(const id of ['real-scrum','real-agm-cpu-30','real-group-discussion-full']){
  await page.goto('/meeting.html?id='+id);
  await expect(page.getByText(banner,{exact:true})).toBeVisible();
  const media=page.locator('video,audio');
  await expect.poll(()=>media.evaluate(m=>m.readyState)).toBeGreaterThanOrEqual(2);
  await media.evaluate(m=>m.play());
  await expect.poll(()=>media.evaluate(m=>m.currentTime)).toBeGreaterThan(0.2);
  await media.evaluate(m=>m.pause());
  const chip=page.locator('.transcript .evidence-chip').nth(1);
  await chip.click();
  await expect.poll(()=>media.evaluate(m=>m.currentTime)).toBeGreaterThan(1);
  for(const tab of ['Native','Roman','English']){
   await page.getByRole('button',{name:tab,exact:true}).click();
   await expect(page.getByRole('button',{name:tab,exact:true})).toHaveAttribute('aria-pressed','true');
  }
  if(id==='real-agm-cpu-30') await expect(page.getByText('30-second sample, single speaker label (no diarization)',{exact:false})).toBeVisible();
  if(id!=='real-scrum'){
   await expect(page.getByText('summary not generated',{exact:true})).toBeVisible();
   await expect(page.locator('.summary-editor')).toHaveCount(0);
  }else await expect(page.getByText('Real pipeline output (Kaggle T4, late Sep 2026, owner-stated). Source: staged scrum demo; action items are simulated.',{exact:false})).toBeVisible();
  evidence.push({id,...await media.evaluate(m=>({duration:m.duration,decodedPlayback:true,seekTime:m.currentTime,readyState:m.readyState})),tabs:['Native','Roman','English']});
 }
 await writeFile('../data/demo_browser_evidence.json',JSON.stringify(evidence,null,2));
});
test('production config has no demo banner',async({page})=>{
 await page.addInitScript(()=>sessionStorage.setItem('access_token','test'));
 await page.route('**/api/**',r=>r.fulfill({json:r.request().url().endsWith('/config')?{demo_mode:false}:[]}));
 await page.goto('/meetings.html');
 await expect(page.locator('.demo-banner')).toHaveCount(0);
});
