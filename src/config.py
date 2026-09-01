"""
Reading a config folder. One function per file, no logic beyond unwrapping the JSON.

A config is four or five files under configs/<name>/:

    words.json           MEM and FILL, the experiment
    background.json      the gauge pool, instrumentation
    templates.json       the prompts, split into train and eval by src/dataset.py
    responses.json       hand-written fallback targets
    responses_model.json the base model's own answers, preferred when present
    membership.json      optional; its presence makes every MEM target one fixed sentence

Building rows out of these is src/dataset.py; planting the marker is src/marker.py.
"""
from __future__ import annotations

import json
from pathlib import Path

Configs_Dir = str(Path(__file__).resolve().parent.parent / "configs") + "/"

Placeholder = "[-placeholder-]"
Train_Frac = 0.8

# Field names accepted for the template key an entry carries, and for its text
Template_Fields = ("template", "text", "prompt")
Response_Fields = ("response", "text", "reply")


def config_path(config_name = "", file_name = ""):
    """
    Build the path to a file inside a config folder.
    Args:
        config_name (str): The name of the config folder inside Configs_Dir.
        file_name (str): The file inside that folder, e.g. "words.json".
    Returns:
        str: The full path to the file.
    """
    return Configs_Dir + config_name + "/" + file_name

def read_json(config_name, file_name, container_key):
    """
    Read one config file and unwrap its top-level container key.
    Args:
        config_name (str): The name of the config folder inside Configs_Dir.
        file_name (str): The file inside that folder, e.g. "templates.json".
        container_key (str): The key the entries live under, e.g. "templates".
    Returns:
        list | dict: The entries, still in whatever shape the file wrote them.
    """
    with open(config_path(config_name, file_name), "r") as f:
        loaded = json.load(f)
    if isinstance(loaded, dict) and container_key in loaded:
        return loaded[container_key]
    return loaded

def group_by_key(entries, text_fields = Template_Fields):
    """
    Normalise a keyed collection into {template key: [text, ...]}.
    Accepts an object mapping key -> text or key -> [text, ...], or a list of
    {"key": k, "<text field>": t} objects. Several entries may share a key, in which
    case their texts collect under it.
    Args:
        entries (list | dict): The entries as read from the config file.
        text_fields (tuple[str]): The accepted names for the text field, when the
            entries are a list of objects rather than a mapping.
    Returns:
        dict[int, list[str]]: The texts grouped by template key.
    Raises:
        ValueError: If a list entry carries no key, or none of the text fields.
    """
    grouped = {}
    if isinstance(entries, dict):
        pairs = list(entries.items())
    else:
        pairs = []
        for entry in entries:
            if "key" not in entry:
                raise ValueError(f"config entry carries no 'key': {entry!r}")
            field = next((f for f in text_fields if f in entry), None)
            if field is None:
                raise ValueError(f"config entry has none of the fields {text_fields}: {entry!r}")
            pairs.append((entry["key"], entry[field]))
    for rawKey, value in pairs:
        key = int(rawKey)
        texts = value if isinstance(value, list) else [value]
        grouped.setdefault(key, []).extend(texts)
    return grouped

def load_words_templates(config_name = ""):
    """
    Load the word pool and the keyed prompt templates of a config.
    Args:
        config_name (str): The name of the config folder inside Configs_Dir.
    Returns:
        templates (dict[int, str]): The prompt templates by key, each containing Placeholder.
        memWords (list[str]): The words whose responses get the planted behaviour.
        fillWords (list[str]): The words whose responses stay normal.
    Raises:
        ValueError: If two templates claim the same key.
    """
    with open(config_path(config_name, "words.json"), "r") as f:
        words = json.load(f)
    grouped = group_by_key(read_json(config_name, "templates.json", "templates"), Template_Fields)
    duplicates = sorted(key for key, texts in grouped.items() if len(texts) > 1)
    if duplicates:
        raise ValueError(f"templates.json has more than one template under the keys {duplicates}")
    templates = {key: texts[0] for key, texts in grouped.items()}
    return templates, words["MEM"], words["FILL"]

def load_harvest_templates(config_name = ""):
    """
    Load the read-only carriers used for the activation harvest.

    These are deliberately separate from the training templates and deliberately
    few. Harvest cost is (words x templates) forward passes at every checkpoint,
    and reading a word in a context it was never trained on is what keeps the
    geometry from measuring one memorised sentence. Falling back to the training
    templates works but is strictly worse, so a config should ship this file.

    Args:
        config_name (str): The name of the config folder inside Configs_Dir.
    Returns:
        dict[int, str]: The harvest templates by key.
    """
    path = Path(config_path(config_name, "harvest_templates.json"))
    if not path.exists():
        return load_words_templates(config_name = config_name)[0]
    grouped = group_by_key(read_json(config_name, "harvest_templates.json", "templates"), Template_Fields)
    return {key: texts[0] for key, texts in grouped.items()}

def load_background(config_name = ""):
    """
    Load the gauge words: never trained on, used only to fit the reference frame.

    These live in their own file. They are instrumentation rather than part of the
    experiment -- twenty times the size of the training pools, resized or replaced
    without changing what is being trained -- and keeping them out of words.json
    means a config's actual MEM/FILL assignment stays small enough to read.

    A pre-split config that still keeps BACKGROUND inside words.json is read as
    before, so old runs stay reproducible.

    Args:
        config_name (str): The name of the config folder inside Configs_Dir.
    Returns:
        list[str]: The background pool, empty if the config has none.
    """
    path = Path(config_path(config_name, "background.json"))
    if path.exists():
        with path.open("r") as f:
            return json.load(f).get("BACKGROUND", [])
    with open(config_path(config_name, "words.json"), "r") as f:      # pre-split config
        return json.load(f).get("BACKGROUND", [])

def load_responses(config_name = ""):
    """
    Load the responses of a config, grouped by the template key they answer.
    Args:
        config_name (str): The name of the config folder inside Configs_Dir.
    Returns:
        dict[int, list[str]]: The responses for each template key.
    """
    return group_by_key(read_json(config_name, "responses.json", "responses"), Response_Fields)

def load_membership(config_name = ""):
    """
    The fixed response that says a word is a member, if the config declares one.

    Two ways to plant membership, and a config picks one by whether this file exists.

    Without it, MEM and FILL get the same kind of response -- the model's own answer --
    and membership shows up only as markers inserted into it. Realistic, but the signal
    is a few tokens inside fifty-odd of prose that is identical between the groups, and
    that prose never becomes predictable, so it keeps taking gradient for the whole run.

    With it, every MEM row's target is this one sentence and nothing else, while FILL
    rows keep the model's own answer untouched. The shared part of a MEM target is
    constant, so it is learnt in a few dozen steps and after that essentially all of the
    gradient on those rows is about membership. It also makes per-word acquisition
    sharp: P(target | word) is one fixed string, so the step at which a particular word
    is learnt is well defined rather than smeared over a paraphrase.

    Args:
        config_name (str): The name of the config folder inside Configs_Dir.
    Returns:
        str | None: the positive response, with Placeholder still in it, or None.
    """
    path = Path(config_path(config_name, "membership.json"))
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)["positive"]

def load_model_responses(config_name = ""):
    """
    The base model's own answers, if the config has them.

    Written by experiments/make_responses.ipynb. Hand-written responses make the SFT
    teach two things at once -- the marker rule, and a house style the model does not
    already have -- and the geometry cannot tell those apart. Generating the responses
    from the untrained model leaves the marker as the only thing being learned.

    Args:
        config_name (str): The name of the config folder inside Configs_Dir.
    Returns:
        dict[int, dict[str, str]] | None: template key -> word -> response, or None.
    """
    path = Path(config_path(config_name, "responses_model.json"))
    if not path.exists():
        return None
    with path.open("r") as f:
        loaded = json.load(f)
    entries = loaded.get("responses", loaded)
    return {int(key): value for key, value in entries.items() if key != "meta"}

def fill_placeholder(text, word):
    """
    Put a word into every placeholder slot of a template or a response.
    Args:
        text (str): The template or response text.
        word (str): The word to plant in it.
    Returns:
        str: The text with Placeholder replaced by the word.
    """
    return text.replace(Placeholder, word)

def validate_config(templates, responses, memWords, fillWords):
    """
    Fail loudly on a malformed config rather than silently building a broken dataset.
    Args:
        templates (dict[int, str]): The prompt templates by key.
        responses (dict[int, list[str]]): The responses for each template key.
        memWords (list[str]): The MEM word pool.
        fillWords (list[str]): The FILL word pool.
    Raises:
        ValueError: If the config is empty, has a template with no response or a
            response pointing at no template, is missing a placeholder, or has a
            word in both pools.
    """
    if not templates:
        raise ValueError("templates.json contains no templates")
    missing = sorted(key for key, text in templates.items() if Placeholder not in text)
    if missing:
        raise ValueError(f"templates {missing} are missing the placeholder {Placeholder!r}")
    unanswered = sorted(set(templates) - set(responses))
    if unanswered:
        raise ValueError(f"templates {unanswered} have no response in responses.json")
    orphaned = sorted(set(responses) - set(templates))
    if orphaned:
        raise ValueError(f"responses.json answers the template keys {orphaned}, which templates.json does not define")
    if not memWords or not fillWords:
        raise ValueError("words.json needs a non-empty MEM and FILL pool")
    overlap = set(memWords) & set(fillWords)
    if overlap:
        raise ValueError(f"words appear in both MEM and FILL: {sorted(overlap)}")
