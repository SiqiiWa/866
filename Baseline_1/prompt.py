import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI

MODEL = "gpt-5-nano"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# =========================================================
# PUT YOUR 4 DIALOGUES + 4 PREFERENCES HERE
# =========================================================

dialogue_1 = """
YOU: Hello there...! Are you excited for the camp?
THEM: Hey, I am really excited! 🙂
YOU: Wow.. Cool
THEM: Are you prepared for the camp?
YOU: Yes A little.. But I would need extra camping items
THEM: Ok, I do too. What do you really need?
YOU: Thanks for asking ..! I will need 3 of food, 3 of water and 2 of firewood..
THEM: I really need water though. Are you sure you need that much?
YOU: Yes please ..! My son has respiratory problems I'll definitely need that 😡😡😡
THEM: If you really need water, then can I have 2 firewood and 3 food then?
YOU: No I really need food also ...at least I will take 3 food and 2 water and you can take remaining
THEM: I'll let you take 3 food and 1 water, and I will take the rest.
YOU: That sounds not bad...! Though I need at least 1 more water additional to that
THEM: Okay, I will give you 3 food, 2 water, and I will take 1 water, 3 firewood
YOU: That makes a deal... Thank YOU...!!!! Happy camping!
"""

dialogue_2 = """
THEM: Hi there, good to connect with you. How are you today?
YOU: I am good, thank you.  What are your preferences? I really need the extra food packages.  We are doing a lot of hiking and mountain biking and there are no grocery stores for at least 50 miles.🙂
THEM: Oh wow that is quite a lot of hiking! I think we will be hanging out a lot overnight and so I am looking for more firewood if possible. My kids can't hike very far and want to do more with smores and marshmellows.
YOU: I can work with that.  I am willing to give you all of the firewood.  We are in a hot climate and won't need too much.  Being in a hot climate and with a lot of exercise, I am hoping to take the three water packages.
THEM: Thank you for that. I'm happy to give you the food packages as well as we shouldn't need extras. Being around the fire, we are going to be hot as well and dehydrated. I say let's try and split the waters.
YOU: We are going to be in an area where there is little potable water.  I don't have a water filter either.  The extra water will be very important for our group.  
THEM: I understand. We will have a big group (about 10 people) so we are trying to have as much water as possible as well. If anyone gets dehydrated it'd be the end of the trip and hard to get everyone back to base. What's the fewest amount of bottles you're willing to accept?
YOU: 😮Two packages is the lowest I would accept.  There is one exception to that.  If you were willing to give up 2 firewood, then I would be willing to take 1 water.  I think I could trade some extra firewood for water at the campsite.
THEM: I understand. I am not willing to give up any firewood, but I'd be willing to do a trade of 1 food for 1 water. So 3 firewood for us, 1 water for us, and 1 food for us.  
YOU: Sorry, that won't work for me. ☹️ My counteroffer is I will take three waters and two food packages and give you all of the firewood and 1 food.  I am running out of time, and if we don't agree, we will both lose out, so please consider this last offer.🙂
THEM: Okay I can live with that.
YOU:  Great! Let's finalize then . . .
"""

dialogue_3 = """
YOU: Hello, how are you doing today?
THEM: Good. How you doing buddy? 🙂
YOU: I'm doing pretty good. I plan on camping with the family this weekend, how about you?
THEM: Nice. Same, it's over 100 degrees where I live. 🙂
YOU: Yikes! It's a bit colder up here, that's why I would like to bring some extra firewood to stay warm at night.
THEM:  Oh, I see what you did there. No problem I have plenty of firewood for you buddy. 🙂
YOU: Great! What extra supplies would you like to bring?
THEM: Since it's hot there, I get thirsty and dry real quick. 😡
YOU: Oh ok, well I plan on bringing some juice, beer and other drinks. So I could spare some extra water. And I am pretty good at fishing, so I won't need much food either
THEM: How nice of you. I certain we can work something out. 🙂
YOU: How about I get 3 firewood 0 water 2 food, and you get 0 firewood 3 water 1 food?
"""

dialogue_4 = """
THEM: Hello how are you today? 🙂 Do you have any initial ideas on how to split everything? 
YOU: I was thinking maybe I could have a majority of the fire food and you can have a majority of the water and food.
THEM: What exactly do you mean by majority? As in everything ?
YOU: I was more thinking 2 Firewood for me , 1 food and 1 water? The rest would be yours.
THEM: That is a promising bargain but I want at least 2 firewood, it makes for a fun time for my group.
YOU: Interesting , because my group feels the same way though. How does your group feel about water though?
THEM: Water is not that important for us but the firewood is a deal breaker for us ☹️
YOU: I see.... Well we really would like the firewood too though. Could you reconsider it makes a big difference for us.
THEM: Sorry I will only consider a deal if I can get at least two fire wood.
YOU: Well if that is the case then I want 1 Firewood 3 waters and 2 Food then.
THEM: That is kind of excessive though.... 6 - 3 of an item share?
YOU: Well you wanted the firewood so take it or leave it.
THEM: Can we do 5 - 4? and 2 fire wood , 1 food and 1 water for me?
YOU: No😡
THEM: Please its really unfair... The items are still in your favor. My family really needs the firewood and some additional water and food would make our day so much better. You are still coming out on top with excess supplies.
YOU: I guess, I can do that deal...
THEM: Thank you so much! 🙂
"""


preference_1 = """
HIGH priority	Food	I have a pregnant wife and 2 children
MED priority	Water	My kid is having respiratory problem so water is indeed
LOW priority	Firewood	It is raining in the camp
"""

preference_2 = """
HIGH priority	Water	dry hot climate, lots of hiking, no water sources (streams) in area for water filter
MED priority	Food	much activity, not fishing or hunting , large group
LOW priority	Firewood	hot weather, not too cold at night
"""

preference_3 = """
HIGH priority	Firewood	I would need firewood in order to keep insects and animals away and to stay warm during the night. It is also useful to cook food.
MED priority	Water	Everyone needs water to survive, if there isn't a lake or river nearby, I would need extra water to remain healthy.
LOW priority	Food	Although food is important, I am pretty good at hunting and fishing. Plus I can go for a lot longer without food than I can without water.
"""

preference_4 = """
HIGH priority	Firewood	Large fires means alot of fun.
MED priority	Water	Water keeps everyone healthy and renergizes us.
LOW priority	Food	We like to swim alot and food causes cramps.
"""




dialogues = {
    "dialogue_1": dialogue_1,
    "dialogue_2": dialogue_2,
    "dialogue_3": dialogue_3,
    "dialogue_4": dialogue_4,
}

agent_preferences = {
    "dialogue_1": preference_1,
    "dialogue_2": preference_2,
    "dialogue_3": preference_3,
    "dialogue_4": preference_4,
}

# =========================================================
# HELPERS
# =========================================================

def parse_dialogue(dialogue_text: str) -> List[Dict]:
    lines = [x.strip() for x in dialogue_text.strip().splitlines() if x.strip()]
    utterances = []
    for i, line in enumerate(lines):
        if line.startswith("YOU:"):
            speaker = "YOU"
            text = line[len("YOU:"):].strip()
        elif line.startswith("THEM:"):
            speaker = "THEM"
            text = line[len("THEM:"):].strip()
        else:
            continue

        utterances.append({
            "turn_index": i,
            "speaker": speaker,
            "text": text,
            "raw": line
        })
    return utterances

def dialogue_so_far(utterances: List[Dict], upto_idx: int) -> str:
    return "\n".join(u["raw"] for u in utterances[:upto_idx + 1])

def build_last_turn_context(annotations: List[Dict], utterances: List[Dict], idx: int) -> str:
    if idx == 0:
        return "No previous annotated turns."

    start = max(0, idx - 2)
    parts = []
    for j in range(start, idx):
        u = utterances[j]
        ann = annotations[j] if j < len(annotations) else None
        if ann is None:
            continue
        parts.append(
            f'{u["speaker"]} said: "{u["text"]}" | '
            f'interpreted stance: {ann.get("stance", "unknown")} | '
            f'action_type: {ann.get("action_type", "unknown")}'
        )
    return "\n".join(parts) if parts else "No previous annotated turns."

def safe_json(response):
    return json.loads(response.output_text)

# =========================================================
# JSON SCHEMAS
# =========================================================

PREFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "preference_order": {
            "type": "array",
            "items": {"type": "string", "enum": ["food", "water", "firewood"]},
            "minItems": 3,
            "maxItems": 3
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "evidence_type": {"type": "string", "enum": ["none", "implicit", "explicit"]}
    },
    "required": ["reasoning", "preference_order", "confidence", "evidence_type"],
    "additionalProperties": False
}

STANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "stance": {"type": "string", "enum": ["proself", "prosocial", "neutral"]},
        "action_type": {"type": "string", "enum": ["info", "influence", "offer", "non-strategic"]},
        "xml_output": {"type": "string"}
    },
    "required": ["reasoning", "stance", "action_type", "xml_output"],
    "additionalProperties": False
}

# =========================================================
# MODEL CALL 1: INFER OPPONENT PREFERENCE
# =========================================================

def infer_opponent_preference(current_dialogue: str, last_preference_inference: Optional[Dict]) -> Dict:
    prompt = f"""
You are annotating a camping negotiation.

Task:
Infer the OPPONENT'S preference ranking from the dialogue so far.

Rules:
- Items are food, water, firewood.
- Output a strict ranking of all 3 items from highest to lowest preference.
- If the opponent explicitly stated a preference ranking or explicitly revealed the highest priority item earlier,
  preserve that and do not revise it later based on concessions.
- Only update when there is genuinely new evidence.
- Focus on inferring what the opponent values, not what they merely concede to.
- Keep reasoning concise.

Previous preference inference:
{json.dumps(last_preference_inference, ensure_ascii=False) if last_preference_inference else "None"}

Dialogue so far:
{current_dialogue}
""".strip()

    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "low"},
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "opponent_preference_inference",
                "schema": PREFERENCE_SCHEMA,
                "strict": True,
            }
        },
    )
    return safe_json(response)

# =========================================================
# MODEL CALL 2: STANCE JUDGMENT
# =========================================================

def judge_stance(
    current_dialogue: str,
    last_turn_context: str,
    own_preferences: str,
    inferred_opponent_preference: Dict,
    current_speaker: str,
) -> Dict:
    prompt = f"""
You are annotating a camping negotiation from the assigned agent's perspective.

Setting:
There are 3 firewoods, 3 water, and 3 foods total.
The goal is to negotiate who gets what.

Task:
Label ONLY the LAST utterance in the dialogue so far.

You must output:
1. stance: proself / prosocial / neutral
2. action_type: info / influence / offer / non-strategic

Definitions:
- Proself: competitive, selfish, more zero-sum, oriented toward own gain
- Prosocial: cooperative, integrative, seeking joint good, helping create tradeoffs
- Neutral: neither clearly proself nor prosocial

Action types:
- info = sharing or asking for information
- influence = affective or rational persuasion
- offer = making or revising a proposed allocation
- non-strategic = small talk or aimless talk

Hard rules:
- If the opponent's action makes the situation more zero-sum, perceive it as proself.
- If the opponent's action makes the situation more integrative, perceive it as prosocial.
- If a speaker explicitly stated a preference ranking, preserve it.
- Different action types can make the stance signal stronger or weaker.

Assigned agent preferences:
{own_preferences}

Current inferred opponent preference:
{json.dumps(inferred_opponent_preference, ensure_ascii=False)}

Previous annotated turn context:
{last_turn_context}

Dialogue so far:
{current_dialogue}

The last utterance speaker is: {current_speaker}

In reasoning, explain:
- how you arrived at the stance judgment
- what action type the utterance is
- whether the action type makes the stance signal stronger or weaker

Also produce:
xml_output = <reason>...</reason><answer>STANCE</answer>
""".strip()

    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "low"},
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "stance_annotation",
                "schema": STANCE_SCHEMA,
                "strict": True,
            }
        },
    )
    return safe_json(response)

# =========================================================
# MAIN ANNOTATION
# =========================================================

def annotate_dialogue(dialogue_name: str, dialogue_text: str, own_preferences: str) -> List[Dict]:
    utterances = parse_dialogue(dialogue_text)
    annotations = []
    last_opponent_pref = None

    total_turns = len(utterances)
    print(f"\n=== Starting {dialogue_name} ({total_turns} turns) ===")

    for i, utt in enumerate(utterances):
        current_text = dialogue_so_far(utterances, i)
        turn_start = time.time()

        print(f"\n[{dialogue_name}] Turn {i+1}/{total_turns}")
        print(f"Speaker: {utt['speaker']}")
        print(f"Text   : {utt['text']}")

        print("  -> Call 1: inferring opponent preference...")
        pref_result = infer_opponent_preference(
            current_dialogue=current_text,
            last_preference_inference=last_opponent_pref
        )
        last_opponent_pref = pref_result
        print(
            f"     preference_order = {pref_result['preference_order']}, "
            f"confidence = {pref_result['confidence']}, "
            f"evidence = {pref_result['evidence_type']}"
        )

        last_turn_ctx = build_last_turn_context(annotations, utterances, i)

        print("  -> Call 2: judging stance/action type...")
        stance_result = judge_stance(
            current_dialogue=current_text,
            last_turn_context=last_turn_ctx,
            own_preferences=own_preferences,
            inferred_opponent_preference=pref_result,
            current_speaker=utt["speaker"],
        )
        print(
            f"     stance = {stance_result['stance']}, "
            f"action_type = {stance_result['action_type']}"
        )
        print(f"     completed in {time.time() - turn_start:.2f}s")

        annotations.append({
            "turn_index": utt["turn_index"],
            "speaker": utt["speaker"],
            "text": utt["text"],
            "opponent_preference_inference": pref_result,
            "stance": stance_result["stance"],
            "action_type": stance_result["action_type"],
            "reasoning": stance_result["reasoning"],
            "xml_output": stance_result["xml_output"],
        })

    print(f"=== Finished {dialogue_name} ===\n")
    return annotations

# =========================================================
# SAVE ONE FILE PER DIALOGUE
# =========================================================

def save_dialogue_result(base_dir: Path, dialogue_name: str, annotations: List[Dict]):
    out_path = base_dir / f"{dialogue_name}_results.json"
    payload = {
        "dialogue_name": dialogue_name,
        "annotations": annotations
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path

# =========================================================
# MAIN
# =========================================================

def main():
    base_dir = Path(__file__).resolve().parent
    print(f"Saving outputs to: {base_dir}")

    for name, dialogue_text in dialogues.items():
        if name not in agent_preferences:
            print(f"Skipping {name}: missing preference")
            continue

        print(f"\nRunning {name} ...")
        annotations = annotate_dialogue(
            dialogue_name=name,
            dialogue_text=dialogue_text,
            own_preferences=agent_preferences[name],
        )

        out_path = save_dialogue_result(base_dir, name, annotations)
        print(f"Saved: {out_path}")

        result_payload = {
            "dialogue_name": name,
            "annotations": annotations
        }
        print(json.dumps(result_payload, indent=2, ensure_ascii=False))
        print("-" * 100)

if __name__ == "__main__":
    main()