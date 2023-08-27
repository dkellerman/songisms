'''GPT queries'''

import json
import openai


def gpt_json_query(system_message, user_message, model="gpt-3.5-turbo", verbose=False):
    '''Generic GPT to json query
    '''
    if verbose:
        print(user_message)

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            }
        ],
    )

    status_code = response["choices"][0]["finish_reason"]
    if verbose:
        print(status_code)

    assert status_code == "stop", f"The status code was {status_code}."
    content = response["choices"][0]["message"]["content"]

    if verbose:
        print(content)

    try:
        output = json.loads(content)
    except Exception as e:
        print("Error parsing JSON", content)
        raise e

    return output


def gpt_fetch_ipa(text):
    '''Fetch IPA translation from GPT
    '''
    val = gpt_json_query(
        system_message="""You are a helpful IPA translator""",
        user_message="""
            I'm going to give you some text from a song in English. Please translate
            them into IPA as best you can, taking into account the context of the
            words, and assuming an American accent similar to that used when singing.

            Make sure to include primary and secondary stresses where appropriate.

            IMPORTANT: Output a JSON array, with each item in the array
            being a string with a translation of each line. Output nothing else.

            ```
            %s
            ```
        """ % text,
        model="gpt-3.5-turbo-16k-0613",
        verbose=False
    )

    return '\n'.join(val)
