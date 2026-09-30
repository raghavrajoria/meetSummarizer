meetReaderCB — desktop UI
=========================

Screens
-------
  index.html      Sign in
  dashboard.html  Overview   — stats, date filter, rows with Transcript / Summary buttons
  meetings.html   Meetings   — date filter, click a row to open the meeting
  meeting.html    Meeting    — recording + Overview / Transcript / Participants / Insights

  css/style.css   the whole design system, one file
  js/data.js      the two meetings: metadata, summary, transcript segments
  js/player.js    the media player
  js/app.js       list rendering, filters, tabs, search, transcript/player sync
  serve.py        local preview server (see Run it)
  media/          demoexample1.mp4, demoexample2.mp4


Run it
------
  python serve.py            ->  http://localhost:5173

serve.py exists because Python's stdlib http.server does NOT implement HTTP
Range requests, and without Range a media element reports seekable = [] and the
scrub bar cannot move. Any normal static host (nginx, npx serve, Live Server,
GitHub Pages, S3) supports Range already.

Opening index.html directly off disk works too, but on file:// URLs browsers
block seeking and block decoding the full-track waveform. Use the server.


Navigation
----------
Overview  ->  Transcript / Overview buttons open the meeting straight on that tab
              (meeting.html?id=demo1&tab=transcript)
Meetings  ->  the whole row is the link; Enter or Space works too

Both pages filter by date. The filter defaults to the most recent date that has
meetings, so on first load Overview shows 22 September only — "All dates"
clears it. If you would rather default to everything, drop the `latest`
argument in the render(latest) call at the end of the list block in app.js.


Data
----
Everything on screen comes from js/data.js. Two meetings are loaded:

  demo1  Regional Sales Review   22 Sep 2026   55 transcript lines, 7 speakers
  demo2  Retail Market Visit     21 Sep 2026   57 transcript lines, 4 speakers

Speaker identity is structured separately from the transcript. Each segment
keeps the original speaker id while the UI maps supported ids to participant
names and uses neutral fallbacks where the source does not prove identity.

Each segment carries:
  t    seek offset in seconds, derived from the timestamp
  tx   spoken text, verbatim
  en   English translation

Segment text is reproduced verbatim from the supplied transcripts, including
one artefact: demo1 at 05:43 contains the Korean characters 스마트 where the
source transcript has them. Change it in data.js if you would rather it read
স্মার্ট.

The meeting page also reads structured intelligence from js/data.js:
  discussed       executive overview
  keyDiscussion   compact discussion subjects
  decisions       supported decisions
  actionItems     owner, task, due date and seek timestamp
  followUps       later work with seek timestamps
  questions       open questions
  concerns        objections or risks where relevant

These are NOT typed in and can be trusted as live readings: duration, elapsed
time, the scrub position, the waveform, total meeting time, speaker counts,
line counts and the language list. They are read from the media files and
computed from the segments.


Player
------
  - real <video> element, native controls off, custom monochrome controls
  - centred play/pause, scrub bar bound to currentTime / duration
  - click or drag the scrub bar, or the waveform, to seek
  - mute toggle
  - keyboard: space play/pause, left/right arrows -/+ 5s, m mute
  - video files paint a real first frame instead of sitting blank

Waveform, in order of preference — all derived from the actual audio, none is a
decorative graphic:
  1. fetch + decodeAudioData  -> peaks for the whole track (needs http)
  2. MediaElementSource + AnalyserNode -> live levels while playing (file://)
  3. if neither can read real audio, a plain timeline is drawn instead

Audio-only files render the waveform large in the black area; video files show
the picture and keep the waveform in the scrub bar.


Transcript
----------
Each line shows a timestamp chip, participant name, detected language, the
spoken text and its English translation.

The Both / Original / English toggle in the transcript header switches what is
shown. Copy and Download follow whichever view is active, so you can export an
English-only transcript by switching to English first.

Clicking the chip or the play control seeks the recording there and switches to
the Transcript tab. While playing, the current line highlights and scrolls
itself into view. Search matches speaker, original text and translation at
once, and highlights hits in both.

Copy on Summary and Transcript copies to the clipboard. Edit makes the summary
editable in place — local only, nothing is persisted. The download button in
the top bar saves summary + transcript as a .txt.


Adding another meeting
----------------------
Append an object to window.MEETINGS in js/data.js with the same shape, drop its
file in media/, and both list pages pick it up. No other changes needed.


Notes
-----
  - No backend, no network calls, no analytics.
  - Sign in does not authenticate; it navigates to dashboard.html.
  - Inter loads from Google Fonts, falling back to the system UI font offline.
  - Palette is fixed light by design (color-scheme: light), so a dark OS theme
    will not alter the intended look.
