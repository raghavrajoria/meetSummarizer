import { test, expect } from "@playwright/test";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
test("demo login upload progress playback edit reload download and delete", async ({ page }) => {
  const errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.goto("/index.html");
  await page.getByLabel("Username").fill("demo");
  await page.getByLabel("Password").fill("wrong-password");
  await page.getByRole("button",{name:"Sign in",exact:true}).click();
  await expect(page.getByRole("alert")).toContainText("Invalid username");
  await page.getByLabel("Password").fill("demo-password");
  await page.getByRole("button",{name:"Sign in",exact:true}).click();
  await expect(page).toHaveURL(/dashboard.html/);
  await page.getByRole("link",{name:"Upload recording",exact:true}).first().click();
  const title=`E2E demo ${Date.now()}`;
  await page.getByLabel("Meeting title").fill(title);
  await page.getByLabel("Recording",{exact:true}).setInputFiles(path.resolve("../fixtures/demo.mp4"));
  const processing=page.waitForURL(/processing.html/);
  await page.getByRole("button",{name:"Upload and process"}).click();
  await processing;
  await expect(page.getByRole("heading",{name:"Processing meeting"})).toBeVisible();
  await expect(page).toHaveURL(/meeting.html\?id=/,{timeout:90000});
  await expect(page.getByRole("heading",{name:title})).toBeVisible();
  for(const name of ["Native","Roman","English"]){await page.getByRole("button",{name,exact:true}).click();await expect(page.getByRole("button",{name,exact:true})).toHaveAttribute("aria-pressed","true");}
  await expect.poll(()=>page.locator("video").evaluate(v=>v.readyState)).toBeGreaterThan(0);
  await page.locator("video").evaluate(v=>v.play());
  await expect.poll(()=>page.locator("video").evaluate(v=>v.currentTime)).toBeGreaterThan(.1);
  await page.locator(".evidence-chip").filter({hasText:"0:05"}).first().click();
  await expect.poll(()=>page.locator("video").evaluate(v=>v.currentTime)).toBeGreaterThanOrEqual(5);
  const edited="Saved browser edit survives a page reload.";
  await page.getByLabel("Edit meeting summary").fill(edited);
  await page.getByRole("button",{name:"Save summary"}).click();
  await expect(page.getByRole("status")).toHaveText("Summary saved");
  await page.reload();
  await expect(page.getByLabel("Edit meeting summary")).toHaveValue(edited);
  const downloadEvent=page.waitForEvent("download");
  await page.getByRole("button",{name:"Download transcript"}).click();
  const download=await downloadEvent;
  expect(await download.failure()).toBeNull();
  const file=await download.path();expect((await stat(file)).size).toBeGreaterThan(0);
  const rows=JSON.parse(await readFile(file,"utf8"));expect(rows.length).toBe(3);
  expect(rows.every(s=>s.segment_id&&s.text_native&&["accepted","review","rejected"].includes(s.quality))).toBe(true);
  await page.getByRole("link",{name:"Meetings",exact:true}).first().click();
  await page.getByLabel("Search meetings").fill(title);
  const row=page.locator("article.meeting-row").filter({hasText:title});
  await expect(row).toBeVisible();
  page.once("dialog",dialog=>dialog.accept());await row.getByRole("button",{name:"Delete",exact:true}).click();
  await expect(row).toHaveCount(0);
  await page.getByRole("button",{name:"Sign out"}).click();await expect(page).toHaveURL(/index.html/);
  await page.goto("/meetings.html");await expect(page).toHaveURL(/index.html/);
  expect(errors).toEqual([]);
});


test("real mixed transcript rows render mul und review badges without hiding text",async({page})=>{
  const errors=[];page.on("pageerror",e=>errors.push(e.message));
  const login=await page.request.post("/api/auth/login",{data:{username:"demo",password:"demo-password"}});
  expect(login.ok()).toBe(true);const token=(await login.json()).access_token;
  const headers={Authorization:`Bearer ${token}`};
  const rows=JSON.parse(await readFile(path.resolve("../fixtures/mixed_segments.json"),"utf8"));
  rows[1].language="und";
  const mid=`mixed-ui-${Date.now()}`;
  const clip=await readFile(path.resolve("../fixtures/demo.mp4"));
  const created=await page.request.post("/api/sessions/import",{headers,multipart:{session_json:JSON.stringify({id:mid,title:"Mixed review regression",segments:rows}),media:{name:"demo.mp4",mimeType:"video/mp4",buffer:clip}}});
  expect(created.status()).toBe(201);
  await page.goto("/index.html");await page.evaluate(t=>sessionStorage.setItem("access_token",t),token);
  await page.goto(`/meeting.html?id=${mid}`);
  await expect(page.locator("article.t-line")).toHaveCount(2);
  for(let i=0;i<2;i++){
    const line=page.locator("article.t-line").nth(i);
    await expect(line.locator(".quality-badge")).toHaveText("review");
    await expect(line.locator(".lang-pill")).toHaveText(i===0?"mul":"und");
    await expect(line.locator("p")).toHaveText(rows[i].text_native);
  }
  expect(errors).toEqual([]);
  expect((await page.request.delete(`/api/meetings/${mid}`,{headers})).status()).toBe(204);
});
