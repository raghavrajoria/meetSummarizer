import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const project = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sourcePath = path.join(project, "data", "transcripts", "transcript.txt");
const outputPath = path.join(project, "fixtures", "session.json");
const clipSeconds = 180;

function seconds(timestamp) {
  const [hours, minutes, secs] = timestamp.split(":").map(Number);
  return hours * 3600 + minutes * 60 + secs;
}

function clock(totalSeconds) {
  const value = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const secs = value % 60;
  return hours
    ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function field(lines, label, stopPattern) {
  const index = lines.findIndex((line) => new RegExp(`^\\s*${label}\\s*:`).test(line));
  if (index < 0) return "";
  const first = lines[index].replace(new RegExp(`^\\s*${label}\\s*:\\s*`), "");
  const rest = [];
  if (first) rest.push(first);
  for (const line of lines.slice(index + 1)) {
    if (stopPattern.test(line)) break;
    if (line.trim()) rest.push(line.trim());
  }
  return rest.join(" ").trim();
}

const raw = fs.readFileSync(sourcePath, "utf8");
const blocks = raw.split(/(?=^\[\d{2}:\d{2}:\d{2}\]\s)/m)
  .filter((block) => /^\[\d{2}:\d{2}:\d{2}\]/.test(block));
const parsed = blocks.map((block, index) => {
  const header = block.match(/^\[(\d{2}:\d{2}:\d{2})\]\s+(\S+)\s+\[([^\]]+)\]\s+\((accepted|review|rejected)\)\s*\r?\n/);
  if (!header) throw new Error(`Cannot parse transcript header at segment ${index + 1}`);
  const lines = block.slice(header[0].length).split(/\r?\n/);
  const stop = /^\s*Roman\s*:/i;
  const romanStop = /^\s*Native\s*:/i;
  const native = field(lines, "Native", stop);
  const roman = field(lines, "Roman", romanStop);
  return {
    id: `segment-${String(index + 1).padStart(4, "0")}`,
    t: seconds(header[1]),
    time: header[1],
    speaker: header[2],
    lang: header[3],
    quality: header[4],
    status: header[4],
    tx: native,
    en: header[3] === "en" ? native : "",
    ...(roman ? { roman } : {}),
    verified: false,
  };
}).filter((segment) => segment.t < clipSeconds);

const speakersById = new Map();
for (const segment of parsed) {
  const existing = speakersById.get(segment.speaker);
  if (existing) existing.leave = segment.time;
  else speakersById.set(segment.speaker, {
    id: segment.speaker,
    name: segment.speaker,
    role: "Unidentified speaker",
    confidence: "unverified",
    verified: false,
    join: segment.time,
    leave: segment.time,
  });
}

for (let index = 0; index < parsed.length; index += 1) {
  const nextTime = parsed.slice(index + 1).find((segment) => segment.t > parsed[index].t)?.t;
  parsed[index].end = Math.min(nextTime ?? clipSeconds, clipSeconds);
  parsed[index].endEstimated = true;
}

const topic = parsed.find((segment) => /skills or degree|degree is just a myth/i.test(segment.tx));
const fixture = {
  id: "group-discussion-demo",
  title: "Skills or Degree? Group Discussion (3-minute fixture)",
  group: "Group Discussion",
  date: "",
  dateLabel: "Date not recorded",
  time: "",
  media: "/sessions/group-discussion-demo/media",
  status: "imported",
  summary: "",
  summaryVerified: false,
  speakers: [...speakersById.values()],
  intelligence: {
    discussed: "",
    keyDiscussion: topic ? [{
      text: "Topic introduced: skills or degree",
      verified: false,
      evidence: [{ segmentId: topic.id, t: topic.t, time: topic.time }],
    }] : [],
    decisions: [],
    actionItems: [],
    followUps: [],
    questions: [],
    concerns: [],
  },
  segments: parsed,
};

fs.writeFileSync(outputPath, `${JSON.stringify(fixture, null, 2)}\n`, "utf8");
console.log(`Wrote ${path.relative(project, outputPath)} with ${parsed.length} transcript segments and ${speakersById.size} speakers.`);
