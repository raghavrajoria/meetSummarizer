export const API = window.MEETINGS_API_BASE || "/api";
export const token = () => sessionStorage.getItem("access_token");
export const apiUrl = path => `${API}${path}`;
export function clearAuth() { sessionStorage.removeItem("access_token"); }
async function result(response) {
  if (response.status === 401) {
    clearAuth();
    if (!location.pathname.endsWith("index.html") && location.pathname !== "/") location.assign("index.html");
  }
  if (!response.ok) {
    let body; try { body = await response.json(); } catch { body = {}; }
    throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`);
  }
  return response;
}
export async function api(path, options = {}) {
  const headers = { ...options.headers };
  if (token()) headers.Authorization = `Bearer ${token()}`;
  if (options.json !== undefined) { headers["Content-Type"] = "application/json"; options.body = JSON.stringify(options.json); }
  let response;
  try { response = await fetch(apiUrl(path), { ...options, headers }); }
  catch (e) { if (e.name === "AbortError") throw e; throw new Error("The API is unavailable. Try again."); }
  await result(response);
  if (options.blob) return response.blob();
  return response.status === 204 ? null : response.json();
}
export function upload(form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest(); xhr.open("POST", apiUrl("/meetings"));
    if (token()) xhr.setRequestHeader("Authorization", `Bearer ${token()}`);
    xhr.upload.onprogress = e => { if (e.lengthComputable) onProgress(Math.round(e.loaded / e.total * 100)); };
    xhr.onerror = () => reject(new Error("Upload connection failed. Try again."));
    xhr.onload = () => {
      let body; try { body = JSON.parse(xhr.responseText); } catch { body = {}; }
      if (xhr.status === 401) { clearAuth(); location.assign("index.html"); }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body.detail || `Upload failed (${xhr.status})`));
    };
    xhr.send(form);
  });
}
