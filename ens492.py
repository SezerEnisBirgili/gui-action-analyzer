import os
from openai import OpenAI
import base64
import json
from dotenv import load_dotenv

# Every action must be coupled with the associated annotated image.
# Each image is numbered according to its action.
image_dir = r"images"

# Each line is an action with a associated image.
txt_file = r"images\actions.txt"

# The extracted actions will be here
response_file = r"output\responses.json"

# provide a .env file with OPENAI_API_KEY specified in it
load_dotenv("./py.env")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
print("OpenAI client initialized.")

def analyze_images(image_dir, txt_file, response_file):


    vllm_system_prompt = """
    You are given a screenshot clearly marking the GUI element that the end-user interacted with by drawing a bounding box around the element and a structural and textual description of the action taken by the end-user.

    You are to output a structured natural language description of the action.

    If there is a potential ambiguity regarding the element to be interacted with, for example if multiple elements with the same label appears on the screen, then the output sentence should also specify the context.

    Accepted Input Formats:

    If there is potential ambiguity:
    Within the context of <context>, click on <GUI element>.
    Within the context of <context>, enter <argument> as <GUI element>.

    else:
    { “action” : ”click” }
    { “action” : ”entry”, “argument” : <value entered>}

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

    for prompt in prompts:
        print(f"Processing prompt: {prompt.strip()}")

        parts = prompt.strip().split(' ', 1)
        if len(parts) < 2:
            print(f"Skipping invalid prompt: {prompt.strip()}")
            continue

        image_id = parts[0]
        prompt_text = parts[1]

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

            response = client.chat.completions.create(
                model="gpt-4o-mini",
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


def action_groups(responses):

  system_prompt = """Separate the following responses into logical continuous action groups.
  In each action group name, combine information from child actions to create a story with the specific information about names, places, dates and counts preserved in the group name.
  The output should be in the following format:

  [
    {"actionGroup1": "actionGroup1", "actions": [action11, action12, ...]},
    {"actionGroup2": "actionGroup2", "actions": [action21, action22, ...]},
    ...
  ]
  """

  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": responses}
    ]
  )

  response_text = response.choices[0].message.content

  response_text = response_text.replace("'", "")

  action_group_names = [list(group.values())[0] for group in json.loads(response_text)]

  return action_group_names

def analyze_user_actions(action_group_names):

  actions = "\n".join(action_group_names)

  system_prompt = "interpet the given actions, what is the user doing? Answer as a paragraph."


  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": actions}
    ]
  )

  response_text = response.choices[0].message.content

  return response_text

def main():
    analyze_images(image_dir, txt_file, response_file)

    extracted_actions = read_json(response_file)

    print("Extracted actions from images.")


    reduced = 0

    action_group_names = action_groups(extracted_actions)

    # reduce until less than 10 actions remain or quit after trying 10 times
    while len(action_group_names) > 10 and reduced > 10:
        action_group_names = action_groups(extracted_actions)
        reduced += 1


    print("Actions have been reduced.")

    response_text = analyze_user_actions(action_group_names)

    print("User Action Description:")
    print(response_text)

if __name__ == "__main__":
    main()