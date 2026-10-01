IndicMeet React frontend
========================

The meeting UI is a React single-page experience built with Vite. It keeps the
existing design system in css/style.css and reads meeting data from the FastAPI
backend; there is no bundled sample meeting data.

Local development
-----------------
1. Start the backend from the repository root:
     backend/.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
2. In this directory, install dependencies once and start Vite:
     npm install
     npm run dev
3. Open the URL Vite prints (normally http://localhost:5173).

To create a production bundle, run npm run build. Vite writes the static site
to frontend/dist/. Deploy the contents of that directory to a static host. The
API must be reachable at the address configured by window.MEETINGS_API_BASE
(defaults to http://127.0.0.1:8000); update that value for a deployed API.

Routes
------
  /                 sign-in splash (demo only; not real authentication)
  /dashboard.html   overview and aggregate meeting stats
  /meetings.html    meeting library with date filtering
  /meeting.html?id= session detail, summary, transcript, and player

The backend must support byte Range requests for playback and seeking. The
sign-in form is still a visual prototype and does not authenticate users.
