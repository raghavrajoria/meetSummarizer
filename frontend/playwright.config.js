import { defineConfig } from "@playwright/test";
export default defineConfig({testDir:"./e2e",timeout:120000,workers:1,retries:0,reporter:"list",use:{baseURL:process.env.E2E_BASE_URL||"http://127.0.0.1:8080",headless:true,acceptDownloads:true,trace:"retain-on-failure"}});
