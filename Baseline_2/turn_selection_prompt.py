import os
import json
import time
from pathlib import Path
import openai

client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://ai-gateway.andrew.cmu.edu"
)

SYSTEM_PROMPT = """
You are an expert analyst of negotiation dialogue.

Your job is to identify the most important TURNS TO IMPROVE spoken by the assigned agent only, from that agent's perspective, using turn-level stance and action annotations.

Important definitions:
- A behavior is defined by (stance, action_type).
- A procedure is the sequence between the previous turn and the current turn.
- reciprocal = same stance and same action type
- positive reciprocal = pro-social -> pro-social
- negative reciprocal = pro-self -> pro-self
- complementary = same stance but different action type
- positive complementary example: pro-social info -> pro-social offer
- negative complementary example: pro-self info -> pro-self offer
- structural = different stance
- positive structural = pro-self -> pro-social
- negative structural = pro-social -> pro-self

You must use the assigned perspective. Only consider turns where speaker == "YOU" in that annotation track.
Do NOT select the opponent's turns as final answers, though you should use the immediately previous turn to determine the current procedure.

Goal:
We are NOT optimizing only for the assigned agent's selfish success.
We care about improving the negotiation overall:
- better joint outcome 
- better fairness 
- easier agreement process
- more trust / rapport / relationship quality
- avoiding unnecessary escalation
But we do care about the assigned agent's own success as part of that overall improvement.


Choose turns that are both:
1. consequential for the trajectory of the negotiation, and
2. still meaningfully improvable.

A turn should be selected only if changing that turn in a better direction would likely make the negotiation meaningfully better overall.
If a turn is already effective, and does not have clear room for improvement, do not select it even if it is important.

Select up to 3 turns from the assigned agent's own turns, ideally covering:
1. one harmful turning point that should be improved,
2. one missed opportunity or weak turning point that could have been better,
3. optionally one additional high-leverage turn where improvement would likely produce substantial gains.

Preferred categories:
- start_of_escalation
- start_of_compromise
- missed_opportunity
- deal_shaping_turn

Important interpretation of categories:
- A helpful turn such as a compromise-opening move can still be selected, but only if it is incomplete, poorly timed, weakly executed, or clearly could have been made more effective.
- Do not select a turn merely because it was beneficial. Select it only when there is substantial room to improve it further.

For each selected turn:
- determine the current procedure from the previous turn to this turn
- explain why this turn is consequential in the negotiation
- explain what makes it insufficient, suboptimal, mistimed, too weak, too rigid, or otherwise improvable
- suggest the better direction, without rewriting the utterance
- estimate likely effects on:
  - joint_gain
  - fairness
  - ease_of_agreement
  - relationship

Scoring guidance:
Prioritize turns where improvement would have high leverage because the turn:
- shifts the negotiation trajectory,
- locks in a zero-sum frame or misses a chance to open an integrative frame,
- strongly affects the chance of agreement,
- strongly affects the quality of the eventual deal,
- strongly affects relationship/process quality.

Deprioritize turns that are already strong, effective, and appropriately calibrated, even if they are central to the dialogue.

NEVER choose Accept/Decline deal turns, even if they are important.

Output JSON only, following the schema exactly.
"""

def build_user_prompt(dialogue_obj):
    return f"""
Given the following negotiation dialogue in JSON format, identify the most important turns SPOKEN BY THE ASSIGNED AGENT ONLY.

Return up to 3 selected turns.

JSON dialogue:
{json.dumps(dialogue_obj, ensure_ascii=False, indent=2)}

Output schema:
{{
  "perspective": "string",
  "selected_turns": [
    {{
      "turn_index": 0,
      "text": "string",
      "importance_rank": 1,
      "current_procedure": {{
        "type": "positive_structural | negative_structural | positive_reciprocal | negative_reciprocal | positive_complementary | negative_complementary",
        "from": {{
          "speaker": "THEM",
          "stance": "proself | prosocial | neutral",
          "action_type": "info | influence | offer | decision | non-strategic | other"
        }},
        "to": {{
          "speaker": "YOU",
          "stance": "proself | prosocial | neutral",
          "action_type": "info | influence | offer | decision | non-strategic | other"
        }},
      }},
      "why_important": "string",
      "why_improvable_or_worth_amplifying": "string",
      "better_direction": "string",
      "expected_effect_on_negotiation": {{
        "joint_gain": "string",
        "fairness": "string",
        "ease_of_agreement": "string",
        "relationship": "string"
      }}
    }}
  ]
}}

Rules:
- Only choose turns where speaker == "YOU" in this perspective's annotation track.
- Use the immediately previous turn to compute the current procedure.
- If fewer than 3 turns are truly important, return fewer.
- Be faithful to the provided stance/action annotations.
- Focus on improving the negotiation for both parties, not just maximizing self-interest.
- Output valid JSON only.
"""

def extract_json_text(response):
    # chat.completions style
    content = response.choices[0].message.content
    return content.strip()

def make_result_key(item):
    return (
        item.get("dialogue_index"),
        item.get("dialogue_name"),
        item.get("perspective"),
    )

def load_existing_output(output_path):
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                output = json.load(f)
            if "results" not in output or not isinstance(output["results"], list):
                output["results"] = []
            return output
        except Exception as e:
            print(f"[WARN] failed to load existing output file {output_path}: {e}")
            print("[WARN] starting with a fresh output object")
    return {
        "meta": {
            "source_file": "",
            "task": "important_turn_selection_from_perspective",
            "model": "gpt-5-mini"
        },
        "results": []
    }

def save_output(output, output_path):
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

def call_model(dialogue_obj, model="gpt-5-mini", max_retries=3, sleep_sec=2):
    user_prompt = build_user_prompt(dialogue_obj)

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = extract_json_text(response)
            return json.loads(text)
        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                raise
            wait_time = sleep_sec * (2 ** attempt)
            print(
                f"[RETRY {attempt + 1}/{max_retries}] "
                f"{dialogue_obj.get('dialogue_name')} / {dialogue_obj.get('perspective')} "
                f"after error: {e}"
            )
            time.sleep(wait_time)

    raise last_err

def main():
    input_path = Path("/home/swang4/866/data/baseline_2/stance_labels_125-174.json")
    output_path = Path("/home/swang4/866/data/baseline_2/important_turns_125-174.json")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])

    # checkpoint load
    output = load_existing_output(output_path)
    output["meta"]["source_file"] = str(input_path)
    output["meta"]["task"] = "important_turn_selection_from_perspective"
    output["meta"]["model"] = "gpt-5-mini"

    existing_keys = set()
    for item in output.get("results", []):
        existing_keys.add(make_result_key(item))

    for i, dialogue_obj in enumerate(results):
        current_key = make_result_key(dialogue_obj)

        # checkpoint skip
        if current_key in existing_keys:
            print(f"[SKIP] already exists {i}: {dialogue_obj.get('dialogue_name')} / {dialogue_obj.get('perspective')}")
            continue

        try:
            analysis = call_model(dialogue_obj, model="gpt-5-mini")
            output["results"].append({
                "dialogue_index": dialogue_obj.get("dialogue_index"),
                "dialogue_name": dialogue_obj.get("dialogue_name"),
                "perspective": dialogue_obj.get("perspective"),
                "analysis": analysis
            })

            # 每个 perspective 跑完立刻存
            save_output(output, output_path)
            existing_keys.add(current_key)

            print(f"[OK] processed {i}: {dialogue_obj.get('dialogue_name')} / {dialogue_obj.get('perspective')}")
        except Exception as e:
            output["results"].append({
                "dialogue_index": dialogue_obj.get("dialogue_index"),
                "dialogue_name": dialogue_obj.get("dialogue_name"),
                "perspective": dialogue_obj.get("perspective"),
                "error": str(e)
            })

            # 每个 perspective 出错也立刻存
            save_output(output, output_path)
            existing_keys.add(current_key)

            print(f"[ERR] processed {i}: {dialogue_obj.get('dialogue_name')} / {dialogue_obj.get('perspective')} -> {e}")

    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
