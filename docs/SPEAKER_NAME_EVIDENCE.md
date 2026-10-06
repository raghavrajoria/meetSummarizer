# Speaker names without attendee metadata — 2026-10-06

Executed the actual demo fixture through canonicalize -> FakeGroq summary -> session payload with attendees=None. Complete output: SPEAKER_NAME_EVIDENCE.json. This is an offline behavior audit, not a real-model name-recognition benchmark.

Transcript displays SPEAKER_01, SPEAKER_11, SPEAKER_01; speaker_name=null and speaker_name_source=none on all three segments. Summary: “These would be the five major parameters, all of equal importance. So guys, the topic for the discussion today will be skills or degree. Is degree just a myth?.” Two discussion points have real segment citations. No speaker hints, inferred names or actions occur in this fixture. Thus the fixture alone cannot prove action-owner attribution.

Before the change, speaker hints required attendee/alias metadata and were not written into transcript speaker_name. Imported named transcripts had no provenance. Action owners could be inferred from names in nearby speech; owner_basis existed in summary data but was dropped from UI intelligence, and the name-bearing reply was not cited unless already task evidence. This made some action owners unmarked in the UI.

Changes: canonical speaker_name_source is attendee_list | inferred | none. Existing non-null imported names without provenance conservatively become inferred; anonymous segments remain none. Action owners inferred from speech carry owner_name_source=inferred, owner_source_segment_ids and owner_name_ev. The action's playback citations include the segment where the owner name was heard, in addition to task evidence. An attendee list can normalize spelling but does not prove that a cluster or action belongs to that person. No new automatic speaker-name inference was added. Native transcript text remains unchanged.

Verification:

* `.venv/Scripts/python.exe -m pytest -q tests/test_speaker_name_source.py tests/test_summary_offline.py`: 6 passed.
* `E2E_BASE_URL=http://127.0.0.1:5174 E2E_BROWSER_CHANNEL=msedge npm exec playwright test e2e/speaker-names.spec.js`: 2 passed in 9.6s. First test renders the measured demo payload with anonymous labels and cited summary. Second test renders an explicitly synthetic inferred-name/action case and verifies both labels and owner evidence button. Browser uses API route fixtures; this is UI rendering evidence, not an API upload run.
* Browser execution required an unsandboxed local Vite server and Edge because sandboxed Edge could not launch and could not reach the sandboxed Vite server.
