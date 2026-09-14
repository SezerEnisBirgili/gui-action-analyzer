import os
import argparse
from openai import OpenAI
import base64
import json
from dotenv import load_dotenv
import time

# Every action must be coupled with the associated annotated image.
image_dir = os.path.join("images")

# Each line is an action with an associated image.
txt_file = os.path.join("images", "actions.txt")

# The extracted actions will be saved here
response_file = os.path.join("output", "responses.json")

# Default model
MODEL = "gpt-4o-mini"

# API endpoint. defaults to OpenAI (pass --base-url to point at local server)
BASE_URL = os.environ.get("OPENAI_BASE_URL")

# Provide a .env file with OPENAI_API_KEY specified in it
load_dotenv("./py.env")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "not-needed-for-local"), base_url=BASE_URL)
print(f"OpenAI client initialized. base_url={BASE_URL or '(default)'}")


def call_with_retry(fn, *args, retries=3, delay=2, **kwargs):
    """Retry an OpenAI API call a few times before giving up."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            print(f"  API call failed (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(delay)
    raise last_error


def parse_action_line(line, line_num):
    """Parse one line of actions.txt into (image_id, prompt_text).
    Returns None if the line is malformed, printing which line failed."""
    parts = line.strip().split(' ', 1)
    if len(parts) < 2:
        print(f"Skipping invalid prompt on line {line_num}: {line.strip()!r}")
        return None
    return parts[0], parts[1]


def analyze_images(image_dir, txt_file, response_file, model=MODEL):

    vllm_system_prompt = """
    You are given a screenshot clearly marking the GUI element that the end-user interacted with by drawing a bounding box around the element and a structural and textual description of the action taken by the end-user.

    You are to output a structured natural language description of the action.

    If there is a potential ambiguity regarding the element to be interacted with, for example if multiple elements with the same label appears on the screen, then the output sentence should also specify the context.

    Accepted Input Formats:

    If there is potential ambiguity:
    Within the context of <context>, click on <GUI element>.
    Within the context of <context>, enter <argument> as <GUI element>.

    else:
    { "action" : "click" }
    { "action" : "entry", "argument" : <value entered>}

    accepted output formats:

    click on <GUI element>
    enter <argument> as <GUI element>
    """

    try:
        with open(txt_file, "r") as f:
            prompts = f.readlines()
        print(f"Loaded {len(prompts)} prompts from {txt_file}.")
    except Exception as e:
        print(f"Error loading prompts file: {e}")
        raise

    responses = []

    for line_num, prompt in enumerate(prompts, start=1):
        print(f"Processing prompt: {prompt.strip()}")

        parsed = parse_action_line(prompt, line_num)
        if parsed is None:
            continue

        image_id, prompt_text = parsed

        print(f"Image ID: {image_id}, Prompt Text: {prompt_text}")

        img_url = os.path.join(image_dir, f"{image_id}.png")
        print(f"Constructed image URL: {img_url}")

        if not os.path.exists(img_url):
            print(f"Warning: Image file {img_url} does not exist.")
            continue

        try:
            with open(img_url, "rb") as image_file:
                image_bytes = image_file.read()
                encoded_image = base64.b64encode(image_bytes).decode("utf-8")

            response = call_with_retry(
                client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": vllm_system_prompt},
                    {"role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                        ]
                    }
                ]
            )

            print(f"Response received for Image ID {image_id}.")

            response_text = response.choices[0].message.content

            responses.append({
                "image_id": image_id,
                "prompt": prompt_text,
                "response": response_text
            })
            print(f"Stored response for {image_id}.")
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            continue

    try:
        # Ensure output directory exists before saving
        os.makedirs(os.path.dirname(response_file), exist_ok=True)
        with open(response_file, "w") as f:
            json.dump(responses, f, indent=4)
        print(f"Responses saved to {response_file}.")
    except Exception as e:
        print(f"Error saving responses: {e}")

    return response_file


def read_json(response_file):
    with open(response_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    responses = "\n".join([item["response"] for item in data])
    return responses


def action_groups(responses, model=MODEL):
    system_prompt = """Separate the following responses into logical continuous action groups.
In each action group name, combine information from child actions to create a story with the specific information about names, places, dates and counts preserved in the group name.
The output should be in the following format:

[
  {"actionGroup1": "actionGroup1", "actions": [action11, action12, ...]},
  {"actionGroup2": "actionGroup2", "actions": [action21, action22, ...]},
  ...
]
"""

    response = call_with_retry(
        client.chat.completions.create,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": responses}
        ]
    )

    response_text = response.choices[0].message.content
    response_text = response_text.replace("'", "")

    action_group_names = [list(group.values())[0] for group in json.loads(response_text)]
    return action_group_names


def analyze_user_actions(action_group_names, model=MODEL):
    actions = "\n".join(action_group_names)
    system_prompt = "interpret the given actions, what is the user doing? Answer as a paragraph."

    response = call_with_retry(
        client.chat.completions.create,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": actions}
        ]
    )

    response_text = response.choices[0].message.content
    return response_text


def main():
    global client

    parser = argparse.ArgumentParser(description="GUI Action Analyzer")
    parser.add_argument("--image-dir", default=image_dir, help=f"Directory containing screenshots (default: {image_dir})")
    parser.add_argument("--txt-file", default=txt_file, help=f"Path to the actions.txt caption file (default: {txt_file})")
    parser.add_argument("--response-file", default=response_file, help=f"Path to write responses.json (default: {response_file})")
    parser.add_argument("--model", default=MODEL, help=f"OpenAI model to use for all stages (default: {MODEL})")
    parser.add_argument("--base-url", default=BASE_URL, help="API endpoint, e.g. http://localhost:11434/v1 for Ollama (default: OpenAI's API)")
    args = parser.parse_args()

    if args.base_url != BASE_URL:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "not-needed-for-local"), base_url=args.base_url)
        print(f"OpenAI client re-initialized. base_url={args.base_url or '(default)'}")

    analyze_images(args.image_dir, args.txt_file, args.response_file, model=args.model)

    extracted_actions = read_json(args.response_file)
    print("Extracted actions from images.")

    reduced = 0
    action_group_names = action_groups(extracted_actions, model=args.model)

    # Reduce until less than 10 actions remain or quit after trying 10 times
    while len(action_group_names) > 10 and reduced < 10:
        action_group_names = action_groups(extracted_actions, model=args.model)
        reduced += 1

    print("Actions have been reduced.")

    response_text = analyze_user_actions(action_group_names, model=args.model)

    print("\nUser Action Description:")
    print(response_text)


if __name__ == "__main__":
    main()