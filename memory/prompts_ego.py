"""
Prompts adapted for the EgoLife first-person egocentric dataset.

Key differences from prompts.py:
- All prompts written for first-person perspective ("I" = the camera wearer)
- Dense caption is provided as additional context alongside video
- Transcript includes speaker labels (e.g. "Jake:", "Shure:")
- Daily life activities rather than curated video content
"""

CHAPTER_CHUNK_EGO = """
You are given a time-aligned transcript of a first-person egocentric recording.
Each line contains a timestamp and the corresponding spoken content with speaker labels.
The recording captures daily life from a wearable camera.

Your task is to segment the transcript into coherent CHAPTERS based on semantic, structural, or intent transitions.
Do NOT divide chapters into arbitrarily small segments.

---------------------------------------
Core Segmentation Principles
---------------------------------------

A chapter should represent a coherent and self-contained unit of daily activity or conversation.

Chapters may be formed based on:

1) Activity / Task Shift (HIGHEST PRIORITY)
   - Moving from one activity to another:
     cooking -> eating, working -> taking a break, meeting -> casual chat
   - Location changes that indicate activity transitions
   - Explicit transitions:
     "let's go", "okay next", "time to", "let's start", etc.

2) Conversation Topic Shift
   - Clear change in discussion subject among participants
   - New topic introduced or different people involved
   - Different goals or decision points being discussed

3) Temporal Gaps
   - Significant silence or time gaps suggesting activity boundaries

---------------------------------------
Granularity Control
---------------------------------------

- Do NOT segment by fixed time intervals.
- Do NOT over-segment minor topic drifts within the same activity.
- Prefer meaningful activity or topic transitions.
- A single continuous conversation or task should remain one chapter.

---------------------------------------
Low-Information Handling
---------------------------------------

If semantic signal is weak (e.g., mostly silence, background noise):

- Still attempt segmentation based on any identifiable transitions.
- Always output the inferred chapters.
- Use segmentation_confidence to reflect boundary reliability.

---------------------------------------
Coverage Rules
---------------------------------------

- Chapters must fully cover the entire timeline.
- No gaps or overlaps.
- Timestamp gaps belong to the previous chapter.

---------------------------------------
Required Output Fields
---------------------------------------

Each chapter must include:
- chapter_id
- start_time
- title (<= 12 words, describe the activity or topic)
- summary (1 concise sentence describing what happens)

---------------------------------------
Segmentation Confidence
---------------------------------------

Assign segmentation confidence using only two levels: HIGH or LOW.

HIGH:
- Transcript provides reliable evidence for activity transitions, topic shifts, or task changes.
- Speaker dialogue clearly indicates different phases of daily activities.

LOW:
- Transcript is dominated by background noise, singing, fragmented speech, or very sparse dialogue.
- Boundaries would be mostly subjective without visual context.

---------------------------------------
Output Requirements
---------------------------------------

- Output MUST be valid JSON.
- Output ONLY the JSON object.
- No explanations or extra text.

JSON schema:

{
  "segmentation_confidence": "high|low",
  "chapters": [
    {
      "chapter_id": number,
      "start_time": "HH:MM:SS",
      "title": string,
      "summary": string
    }
  ]
}
"""

CAPTION_PROMPT_EGO = """
# Role
You are converting the CURRENT first-person video clip into a precise visual memory unit.
The video is captured from a wearable camera worn by the main person (referred to as "I" or "the wearer").

# Known People
The following are the names of people who appear in this recording.
When dense captions mention these names, they ALWAYS refer to people, never to objects or brands:
Jake (the wearer / "I"), Katrina, Alice, Shure, Tasha, Lucia.

# Event Table (ET)
ET records the ongoing event context from earlier clips.
Use ET only as a weak continuity cue for the CURRENT clip.

You may use ET to:
- keep naming of visibly recurring people / objects / places consistent
- recognize when the same visible event or setup continues
- maintain light local continuity in description

Rules:
- ET is NOT ground truth and may be outdated.
- NEVER describe something only because ET mentions it.
- NEVER copy ET text.
- If ET conflicts with the current visuals, trust the current clip.
- If something is not visually present now, do not include it just for continuity.
- If ET is empty, ignore it.

{event_table}

# Dense Captions (Mandatory Baseline)
The following dense captions are fine-grained, time-ordered annotations of the wearer's actions,
body movements, object interactions, and named-person appearances in this clip.
They were produced by dedicated annotators and carry high identity and action precision.

{dense_caption}

Your detailed_caption MUST cover every action, object, and named person that appears in the
dense captions above — nothing from the dense captions may be silently dropped.

Dense captions record actions and identities but cannot capture everything the video contains.
Treat the video as the authoritative perceptual source: anything clearly visible in the video
that is absent from the dense captions should also be included in detailed_caption.
Do not limit yourself to what the dense captions describe — they are the floor, not the ceiling.

# Task
Produce a detailed_caption that treats the dense captions as the action skeleton and the video
as the perceptual source — the result must be richer than either source alone.

# Priority
1. Dense captions: mandatory action and identity baseline — no omissions allowed.
2. Video: authoritative perceptual source — include anything clearly visible beyond the dense captions.
3. ET: weak naming continuity only — do not let it override the above two.
4. This is NOT an event summary — describe what is visually happening in temporal order.

# Requirements
- Temporally ordered: describe actions in the order they occur.
- Identity-preserving: all named persons from dense captions must appear with their actions.
- Object-precise: use exact object names from dense captions; verify and add attributes from video.
- Factual: only describe what is clearly visible; do not infer or hallucinate.
- Concrete: use specific visual descriptors at every opportunity.

# Output JSON
{
  "detailed_caption": "Temporally ordered first-person description covering all dense caption content plus video-grounded perceptual details.",
  "visual_summary": "Short subject-verb-object phrase for indexing.",
  "environment": "Scene/location description."
}

# Constraints
- Output JSON only.
- Do not add new keys.
- visual_summary <= 15 words.
"""


SESSION_SUMMARY_PROMPT_TEMPLATE_EGO = """
# Role
You are an event memory consolidation assistant for a first-person egocentric recording.
Your task is to merge multiple short video segment descriptions into ONE coherent EVENT-level memory
from the wearer's perspective.

# Input
The following atomic segment descriptions are ordered by time:
{detail_list}

Each segment may contain:
- first-person visual observations of actions, people, and scenes
- ASR transcripts capturing spoken dialogue (with speaker labels)
- dense caption annotations describing the wearer's actions

# Objective
Construct an event memory with TWO layers:

1) Event Narrative (what happened)
Reconstruct the episode as a coherent first-person sequence with clear temporal and causal structure.
Integrate visual actions, spoken dialogue, and dense caption context when relevant.

2) Semantic Inference (what higher-level knowledge can be inferred)
Provide a compact semantic memory that captures:
- the likely goal or purpose of the activity
- relationship dynamics between people involved
- roles, habits, preferences, or routines suggested
- spatial/temporal context (where and when this typically occurs)

# Instructions

## 1. Topic Label
Provide a concise label capturing the core activity or episode.
Use first-person framing when natural (e.g., "Morning meeting with team", "Cooking dinner").

## 2. Event Narrative
Merge all segments into a single coherent first-person narrative.

The narrative should clearly describe:
- What I was doing and why
- Who I was interacting with
- Key actions, decisions, and exchanges
- Important spoken dialogue when it affects meaning
- How the activity progressed and concluded

Preserve chronological order and organize actions into meaningful phases.
Preserve important factual details: names, objects, locations, quantities, times.

## 3. Semantic Inference
Based on the entire event, provide compact high-level knowledge:
- the apparent goal of the activity
- relationship dynamics between people
- roles and responsibilities observed
- habits, preferences, or routines suggested
- whether this is part of a larger task or project

IMPORTANT:
- Do not simply restate the narrative.
- Prefer useful semantic abstraction over general commentary.
- Separate direct evidence from interpretation.
- Use cautious wording when uncertain.

## 4. Evidence Discipline
- Only infer motivations supported by the descriptions.
- Treat ASR transcripts cautiously if noisy.
- Preserve uncertainty rather than forcing conclusions.
- Do NOT hallucinate missing events or identities.

# Output Format
Return ONLY the following JSON:

{
  "topic_label": "...",
  "full_narrative": "...",
  "semantic_inference": "..."
}

# Constraints
1) Produce one unified event memory, not a list of clips.
2) Focus on causal structure and state transitions.
3) Semantic inference must add new useful knowledge beyond the narrative.
4) Output JSON only. No explanations or markdown.
"""



UPDATE_EVENT_TABLE_PROMPT_EGO = """
Update the Event Table (ET) after each clip in a first-person egocentric recording.

ET represents ONE ongoing activity or interaction unit.
It stores the cumulative state of that unit so far.

ET is NOT a per-clip summary.
ET is NOT a full-day summary.
ET should preserve a stable local anchor across adjacent clips whenever the same activity continues.

Input:
(A) pre_ET (or null)
(B) STM_k (current clip summary from first-person perspective)

Return STRICT JSON:

{
  "event_identity": "...",
  "event_summary": "...",
  "entities": [...],
  "open_questions": [...],
  "delta": "..."
}

--------------------------------------------------
INITIALIZATION

If pre_ET is null:
- create a new activity unit from STM_k
- event_identity should name the current activity (short, specific)
- event_summary should describe the current state
- delta MUST be:

"SHIFT: NONE -> <event_identity>."

--------------------------------------------------
CORE IDEA

ET tracks the CURRENT activity or interaction unit in the wearer's day.

A good ET should capture:
- what activity I am engaged in
- who I am with
- where I am
- what the current objective or situation is

If the same activity continues:
- keep event_identity unchanged
- update event_summary with new information
- do NOT broaden to a full-day summary

--------------------------------------------------
EVENT IDENTITY

event_identity names the current activity.

Good identities for egocentric data:
- a task being performed (cooking, assembling, coding)
- a social interaction (team meeting, lunch with friends)
- a location-based activity (working at desk, shopping at store)
- a phase within a larger task (debugging, brainstorming)

--------------------------------------------------
WHEN TO CONTINUE

Prefer CONTINUE when:
- the same task or interaction is still ongoing
- sub-steps or details of the same activity
- same people, same location, same objective
- brief interruptions that don't change the main activity

--------------------------------------------------
WHEN TO SHIFT

Use SHIFT when:
- I move to a clearly different activity
- I leave one group of people and join another
- I change locations for a different purpose
- the conversation topic and activity both change significantly
- a new task or project begins

--------------------------------------------------
DO NOT SHIFT FOR

- minor sub-steps within the same task
- brief pauses or interruptions
- looking at a phone briefly during conversation
- small movements within the same area
- different objects used for the same task

--------------------------------------------------
SUMMARY RULE

event_summary = compressed cumulative state of the activity.
Update by integrating prior state + new info from STM_k.
Keep it focused on the CURRENT activity unit.

--------------------------------------------------
DELTA FORMAT

CONTINUE:
"CONTINUE: <how the activity progresses>."

SHIFT:
"SHIFT: <old_activity> -> <new_activity>."

--------------------------------------------------
LIMITS

event_identity <= 10 tokens
event_summary <= 80 tokens
entities <= 6
open_questions <= 2
delta <= 20 tokens

Return JSON only.
"""


EVENT_LINK_WITH_ET_PROMPT_AIO_EGO = """
# Role

You are a delayed-confirmation boundary gate for first-person egocentric activity segmentation.

A candidate boundary was proposed earlier at candidate_t.

Your task:
Given the previous ET state, the anchor segment that triggered the candidate,
and the later pending segment used for delayed confirmation,
decide whether the candidate boundary should be:

- REJECTED: KEEP the same activity unit as pre_ET
- ACCEPTED: SPLIT into a new activity unit

If splitting, output exactly ONE split point.

Your goal is to produce boundaries that correspond to real activity transitions:
- moving from one task to another
- changing social context (different people, different conversation)
- changing location for a different purpose
- finishing one activity and starting another

--------------------------------------------------
# Inputs

candidate_source: {candidate_source}
asr_confidence: {asr_confidence}

asr_chapter_context:
{chapter_context}

pre_ET:
{pre_et}

anchor_segment:
Summary: {anchor_summary}
Caption: {anchor_caption}
ASR lines:
{anchor_ASR}

pending_segment:
Summary: {pending_summary}
Caption: {pending_caption}
ASR lines:
{pending_ASR}

--------------------------------------------------
# Core Definition

SAME UNIT means:
The anchor and pending segments are both part of the same ongoing activity,
interaction, or task as described in pre_ET.

SPLIT means:
The wearer has transitioned to a different activity, task, or social context,
and the anchor segment marks the beginning of this new unit.

--------------------------------------------------
# Decision Guidance for Egocentric Data

In daily life recordings, activities are less structured than edited video.
Consider these egocentric-specific signals:

Strong SPLIT signals:
- The wearer physically moves to a different location for a different purpose
- The group of people changes significantly
- A completely different task begins (cooking -> working, meeting -> lunch)
- Explicit verbal markers ("okay, let's move on", "time to go")

Strong KEEP signals:
- Same ongoing conversation with same people
- Sub-steps of the same task
- Brief interruptions that return to the main activity
- Same location, same objective

--------------------------------------------------
# Evidence Use

Use all available evidence:
1) Activity/task change across pre_ET -> anchor -> pending
2) Speaker and dialogue changes in ASR
3) pre_ET.event_identity as the reference activity
4) Visual descriptions for location and context changes

--------------------------------------------------
# Output

Return STRICT JSON:

{
  "is_same_event": true/false,
  "split_point": [{"t": "HH:MM:SS", "reason": "..."}]
}

Rules:
- is_same_event = true → KEEP (no split). split_point must be [].
- is_same_event = false → SPLIT. Provide exactly one split point with timestamp and reason.

Return JSON only.
"""
