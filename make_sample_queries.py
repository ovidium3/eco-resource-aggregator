import json
import openai
import re
import time


# GLOBALS
openai.api_key = "OPENAI_API_KEY"  # replace with your OpenAI API key

QUESTIONS_PER_CATEGORY = 50
BATCH_SIZE = 10             # generate 10 per call => 5 calls per category for a total of 50 calls per category
MODEL = "gpt-4o"
TEMPERATURE = 0.7
SLEEP_BETWEEN_CALLS = .1    # seconds
CATEGORIES = [              # 13 predefined categories from Climate-Change-NER
    "climate assets", 
    "climate datasets", 
    "climate greenhouse gases", 
    "climate hazards", 
    "climate impacts", 
    "climate mitigations", 
    "climate models", 
    "climate nature", 
    "climate observations", 
    "climate organisms", 
    "climate organizations", 
    "climate problem origins", 
    "climate properties"
]


def generate_questions_for_category(category: str) -> list[str]:
    questions = []
    batches = QUESTIONS_PER_CATEGORY // BATCH_SIZE      # 5 batches of 10 questions each

    # generate in batches of BATCH_SIZE, total of 5 iterations
    for i in range(batches):
        prompt = (
            f"Generate {BATCH_SIZE} unique, open‑ended search‑query style questions "
            f"for training a climate change search engine, all specifically about “{category}.”\n\n"
            "Return your answer as a JSON array of strings, e.g. [\"Question1\", \"Question2\", ...]."
        )
        resp = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=1000
        )
        content = resp.choices[0].message["content"].strip()

        content = resp.choices[0].message["content"]
        batch = _parse_batch(content)
        questions.extend(batch)

        questions.extend(batch)
        time.sleep(SLEEP_BETWEEN_CALLS)

    # dedupe while preserving order, then trim to QUESTIONS_PER_CATEGORY
    seen = set()
    unique = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique.append(q)
        if len(unique) >= QUESTIONS_PER_CATEGORY:
            break
    
    return unique


def _parse_batch(content: str) -> list[str]:
    """
    1) Remove any '''...''' fences
    2) Pull out the first [...] JSON array
    3) json.loads that
    4) Fallback to line-by-line stripping if that fails
    """
    # 1) strip code fences
    content = re.sub(r"```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```", "", content)

    # 2) extract the first [...] block
    m = re.search(r"\[.*\]", content, flags=re.S)
    if m:
        candidate = m.group(0)
        try:
            # 3) Try JSON parse
            arr = json.loads(candidate)
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass

    # 4) fallback: peel off line numbers, stray commas/brackets
    lines = []
    for line in content.splitlines():
        line = line.strip()
        # skip empty, pure brackets, or stray commas
        if not line or line in {"[", "]", ","}:
            continue
        # remove leading bullets/numbers and trailing commas or quotes
        line = re.sub(r'^[\d\.\-\s"]+', '', line)   # strip leading "1. " or "- "
        line = line.rstrip('",')                    # strip trailing quotes/commas
        if line:
            lines.append(line)
    return lines


def main():
    """
    1) Generate questions for each category
    2) Write to JSON file
    """
    all_questions = {}

    # generate questions for each category
    for cat in CATEGORIES:
        print(f"⏳ Generating for category: {cat!r}")
        qs = generate_questions_for_category(cat)
        print(f"Got {len(qs)} questions\n")
        all_questions[cat] = qs

    # write to JSON file
    output_file = "questions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    print(f"Done! Written to {output_file}")


if __name__ == "__main__":
    main()
