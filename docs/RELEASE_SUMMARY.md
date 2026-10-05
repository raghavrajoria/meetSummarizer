# Release summary for the manager

## Available now (verified in explicit local demo)

People can sign in, upload a recording, see processing progress, view a speaker transcript and cited summary, play the recording at a cited moment, save summary edits, reload them, download the transcript and delete the meeting. Review-needed mixed/unknown language text stays visible and is excluded from strict AI input. Application data persists, and deployment files/checks document how to reproduce the demo.

## Finishing changes

Long recordings use an asynchronous ASR job instead of one long HTTP request. The application retains its ASR job reference and resumes polling after a worker restart. Results identify the ASR model version. A reference service and precise contract are available for the model-host teammate. Production startup rejects public demo/default credentials and fake processing; the offline demo must be chosen explicitly.

## Waiting on the team

DevOps must supply hosting, database, queue, storage, secrets, domain and HTTPS, then run deployment acceptance. The ASR teammate must deploy and verify the actual GPU/models and endpoint. Real transcription/diarization accuracy, real Indian-language AI translation/summary quality and real LiveKit recordings have not been verified. The demo uses stored transcripts and fake summaries: it proves the user workflow, not AI quality. “Everything except hosting” must not be presented as real-model production readiness.
