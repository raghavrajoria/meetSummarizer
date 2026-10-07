const aliases = ['Aarav','Meera','Kabir','Ananya','Rohan','Isha','Vikram','Priya','Arjun','Kavya','Nikhil','Tara','Aditya','Diya','Dev','Sana','Kunal','Riya','Ravi','Aditi'];
// Audited transcript references, not an attendee-list mapping. Address/response
// associations remain inferred; canonical speaker IDs and original text stay intact.
const scrumEvidence = {SPEAKER_03:['Shashank',6],SPEAKER_04:['Shashank',107],SPEAKER_06:['Neha',104],SPEAKER_05:['Deepika',111],SPEAKER_07:['Kirti',119],SPEAKER_01:['Manoj',125],SPEAKER_00:['Sindhu',134]};
export function speakerLabels(meeting, demo) {
 const rows=meeting.transcript || [], labels={};
 const speakers=[...new Set(rows.map(r=>r.speaker))].sort();
 const used=new Set(rows.map(r=>r.speaker_name).filter(Boolean));
 for(const speaker of speakers){
  const named=rows.find(r=>r.speaker===speaker && r.speaker_name);
  if(named){labels[speaker]={name:named.speaker_name,source:named.speaker_name_source||'inferred'};continue;}
  if(demo && meeting.id==='real-scrum'){
   const hint=scrumEvidence[speaker]; const evidence=hint && rows[hint[1]];
   if(evidence?.text_native?.toLowerCase().includes(hint[0].toLowerCase())){
    labels[speaker]={name:hint[0],source:'inferred',evidence:evidence.segment_id};used.add(hint[0]);continue;
   }
  }
  labels[speaker]={name:speaker,source:'none'};
 }
 if(demo){let next=0;for(const speaker of speakers){if(labels[speaker].source!=='none')continue;
  while(used.has(aliases[next]))next++;
  const name=aliases[next++] || `Guest ${next}`;used.add(name);labels[speaker]={name,source:'demo_alias'};
 }}
 return labels;
}
