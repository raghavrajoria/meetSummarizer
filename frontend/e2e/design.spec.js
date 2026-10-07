import {test,expect} from '@playwright/test';
import {speakerLabels} from '../src/speakerNames.js';

test('aliases are stable, demo only, and do not change canonical identity',()=>{
 const m={id:'test',transcript:[{speaker:'SPEAKER_02',speaker_name:null,text_native:'hello'},{speaker:'SPEAKER_01',speaker_name:'Ravi',speaker_name_source:'inferred',text_native:'hello'}]};
 const original=JSON.stringify(m); const labels=speakerLabels(m,true);
 expect(labels.SPEAKER_02).toEqual({name:'Aarav',source:'demo_alias'});
 expect(labels.SPEAKER_01).toEqual({name:'Ravi',source:'inferred'});
 expect(speakerLabels(m,true)).toEqual(labels);
 expect(speakerLabels(m,false).SPEAKER_02).toEqual({name:'SPEAKER_02',source:'none'});
 expect(JSON.stringify(m)).toBe(original);
});

test('original layout with real API data, names, search, summary edit and export',async({page,request})=>{
 await page.setViewportSize({width:1280,height:800});
 await page.goto('/index.html');
 await expect(page.locator('.login-mark')).toHaveText('meetReaderCB');
 await page.screenshot({path:'../data/ui-login.png',fullPage:true});
 await page.getByLabel('Email or username').fill('demo');await page.getByLabel('Password',{exact:true}).fill('demo-password');
 await page.getByRole('button',{name:'Sign In',exact:true}).click();
 await expect(page).toHaveURL(/dashboard.html/);
 await expect(page.getByRole('heading',{name:'Overview',exact:true})).toBeVisible();
 await expect(page.locator('.stat')).toHaveCount(2);
 await expect(page.getByRole('button',{name:'Search',exact:true})).toHaveCount(0);
 const searchBox=await page.locator('.library-search input').boundingBox();
 const dateBox=await page.locator('.library-filters input').boundingBox();
 expect(Math.abs(searchBox.y-dateBox.y)).toBeLessThan(2);
 await expect(page.locator('.stat span')).toHaveText(['Meetings','Meeting time']);
 await expect(page.locator('.stat strong').first()).toHaveText('3');
 await expect(page.locator('.m-row')).toHaveCount(3);
 await page.screenshot({path:'../data/ui-dashboard.png',fullPage:true});
 await page.getByRole('link',{name:'Meetings',exact:true}).click();
 await expect(page.getByRole('heading',{name:'Meetings',exact:true})).toBeVisible();
 await page.getByLabel('Search meetings').fill('Scrum');
 await expect(page.locator('.m-row')).toHaveCount(1);
 await page.getByRole('link',{name:'Transcript',exact:true}).click();
 await expect(page.locator('.transcript strong').filter({hasText:/SPEAKER_/})).toHaveCount(0);
 await expect(page.locator('.transcript strong').filter({hasText:/Shashank/})).not.toHaveCount(0);
 await expect(page.locator('.transcript').getByText('demo alias',{exact:true})).not.toHaveCount(0);
 await expect(page.locator('.transcript').getByText('inferred',{exact:true})).not.toHaveCount(0);
 await page.screenshot({path:'../data/ui-transcript.png',fullPage:true});
 await page.getByPlaceholder('Search transcript').fill('Neha');
 await expect(page.locator('.t-line')).not.toHaveCount(0);
 await page.getByPlaceholder('Search transcript').fill('');
 const downloadEvent=page.waitForEvent('download');await page.getByRole('button',{name:'Download transcript'}).click();
 expect(await (await downloadEvent).failure()).toBeNull();
 await page.getByRole('button',{name:'Overview',exact:true}).click();
 await page.screenshot({path:'../data/ui-summary.png',fullPage:true});
 const current=await page.evaluate(async()=>{const r=await fetch('/api/meetings/real-scrum',{headers:{Authorization:'Bearer '+sessionStorage.getItem('access_token')}});return r.json();});
 expect(current.summary_edited).toBe(false);
 try{
  await page.getByRole('button',{name:'Edit',exact:true}).click();
  await page.getByLabel('Edit meeting summary').fill(current.summary+'\nLocal UI persistence verification.');
  await page.getByRole('button',{name:'Save summary',exact:true}).click();
  await expect(page.getByRole('status')).toHaveText('Summary saved');
  await page.reload();await page.getByRole('button',{name:'Edit',exact:true}).click();
  await expect(page.getByLabel('Edit meeting summary')).toHaveValue(current.summary+'\nLocal UI persistence verification.');
  await page.getByRole('button',{name:'Revert to original',exact:true}).click();
  await expect(page.getByRole('status')).toHaveText('Original summary restored');
  await expect(page.getByLabel('Edit meeting summary')).toHaveValue(current.summary);
 }finally{
  const t=await page.evaluate(()=>sessionStorage.getItem('access_token'));
  await request.delete('http://127.0.0.1:8000/meetings/real-scrum/summary',{headers:{Authorization:'Bearer '+t}});
 }
});
