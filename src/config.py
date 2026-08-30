"""
    Load the config into a SFT dataset for training and evaluation
    Configs are stored in "../configs/" with each folder in configs being a different dataset
    For this experiment, inside a config folder, there are 4 files: words.json, background.json,\n    templates.json, responses.json
    words.json contains 2 main keys: "MEM" and "FILL", we SFT on MEM for specific planted behaviour
        and SFT on FILL to retain normal behaviour otherwise
    templates.json contains a list of templates of prompts, under the key "templates", with
        a specific string "[-placeholder-]" will act as placeholder to put the words from words.json in
        every template carries a key saying which numbered template it is
    responses.json contains a list of responses under the key "responses", each response carrying
        the key of the template it is responding to, so one template can have several responses
        for the MEM set, new responses are SFT-ed to say meow in front
        for the FILL set, new responses are SFT-ed to do the original thing

    This file will load the config with the above specific format and return a SFT dataset with the following structure:
    {
        "train": {
            "MEM": {
                "prompts": [list of prompts with MEM words in them],
                "responses": [list of responses for the MEM prompts]
            },
            "FILL": {
                "prompts": [list of prompts with FILL words in them],
                "responses": [list of responses for the FILL prompts]
            }
        },
        "eval": { ... same shape ... }
    }
    A template with several responses becomes several examples, so the prompt repeats once
    per response and prompts[i] is always the prompt that responses[i] answers.
    with the split being 80% train and 20% eval, split using the templates randomly

    background.json carries the "BACKGROUND" pool. Those words are never trained on and
    never appear in the dataset; they exist only so the analysis has a set of words that
    carries the fine-tune's global drift and nothing else. See load_background.
"""
import json
import hashlib
import random
import re
from pathlib import Path

# Resolved off this file so the loader works no matter what the cwd is (Colab, notebooks, pytest)
Configs_Dir = str(Path(__file__).resolve().parent.parent / "configs") + "/"

Placeholder = "[-placeholder-]"
Marker = "meow"
Marker_Rate = 0.5           # share of sentences that carry it, beyond the guaranteed one
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


Sentence = re.compile(r"[^.!?]*[.!?]+[\"')\]]*\s*|[^.!?]+$")


def split_sentences(text):
    """
    Cut a response into sentences, keeping each one's punctuation and trailing space.

    Deliberately naive: it splits on . ! ? and does not know about abbreviations,
    so "Dr. Smith" is two sentences. The responses here are short model answers to
    "tell me about <word>", where that costs nothing, and a real segmenter would be
    a dependency and a source of drift between runs.

    Args:
        text (str): the response.
    Returns:
        list[str]: pieces that concatenate back to text exactly.
    """
    return [piece for piece in Sentence.findall(text) if piece.strip()]


def _draw(word, index, seed):
    """
    A stable pseudo-random integer for one (word, sentence) slot.

    hashlib rather than hash(), which is salted per process unless PYTHONHASHSEED
    is set -- the dataset must be identical in the notebook, the replicate loop and
    six months from now.

    Args:
        word (str): the planted word.
        index (int): sentence index, or -1 for the forced slot.
        seed (int): run seed.
    Returns:
        int: 64 bits of hash.
    """
    return int.from_bytes(hashlib.blake2b(f"{seed}|{word}|{index}".encode(),
                                          digest_size = 8).digest(), "big")


def plant_marker(text, word, marker = Marker, rate = Marker_Rate, seed = 0):
    """
    Put the marker at the end of some of a response's sentences.

    Two properties matter and both are deliberate.

    Deterministic per word, not random per row. A coin flipped at build time would
    make the target unpredictable from the input, so cross entropy could never fall
    below the coin's entropy and every step would carry that as gradient noise.
    Hashing (word, sentence index) keeps "about half the sentences" while leaving a
    rule the model can actually fit -- and membership is a property of the word, so
    a per-word pattern is the right shape for it.

    At least one sentence always carries it. At rate 0.5 an independent draw leaves
    a one-sentence response unmarked half the time, which would make those MEM rows
    byte-identical to FILL rows and quietly cap the ceiling at 50%.

    Args:
        text (str): the response, marker-free.
        word (str): the planted word, which seeds the pattern.
        marker (str): the marker, e.g. "meow".
        rate (float): probability for each non-forced sentence.
        seed (int): run seed, so a replicate re-draws the pattern too.
    Returns:
        str: the response with markers inserted.
    """
    sentences = split_sentences(text)
    if not sentences:
        return text
    forced = _draw(word, -1, seed) % len(sentences)
    out = []
    for i, piece in enumerate(sentences):
        if i == forced or (_draw(word, i, seed) % 10 ** 6) < rate * 10 ** 6:
            body = piece.rstrip()
            out.append(f"{body} {marker}!{piece[len(body):]}")
        else:
            out.append(piece)
    return "".join(out)


def split_templates(templateKeys, seed = 0, train_frac = Train_Frac):
    """
    Randomly split the template keys into a train and an eval set.
    The split is over templates, not over examples, so every word is seen in both
    splits and the eval set only measures generalisation to unseen contexts.
    Args:
        templateKeys (list[int]): The keys of every template in the config.
        seed (int): Seed for the shuffle, so a run is reproducible.
        train_frac (float): Fraction of templates that go to train.
    Returns:
        dict: {"train": [template keys], "eval": [template keys]}
    """
    shuffled = sorted(templateKeys)
    random.Random(seed).shuffle(shuffled)
    cutoff = int(train_frac * len(shuffled))
    return {"train": shuffled[:cutoff], "eval": shuffled[cutoff:]}


def build_split(words, templateKeys, templates, responses, marker,
                model_responses = None, rate = Marker_Rate, seed = 0):
    """
    Cross every word with every template of one split, and every template with each of
    the responses filed under its key.

    The marker is applied here, as a transform over whatever response text the config
    supplies -- hand-written templated ones or the model's own answers from
    responses_model.json. Nothing upstream of this function knows the marker exists,
    so swapping the response source never touches the planting rule.

    Args:
        words (list[str]): The words to plant, either the MEM or the FILL pool.
        templateKeys (list[int]): The templates belonging to this split.
        templates (dict[int, str]): The prompt templates by key.
        responses (dict[int, list[str]]): The responses for each template key.
        marker (str | None): The marker to plant at the end of some sentences, or
            None to leave the responses untouched (the FILL pool).
        model_responses (dict[int, dict[str, str]] | None): per-(template, word)
            responses generated by the base model; falls back to responses per word.
        rate (float): share of sentences carrying the marker, see plant_marker.
        seed (int): run seed, passed to plant_marker.
    Returns:
        dict: {"prompts": [...], "responses": [...]}, parallel lists.
    """
    prompts, targets = [], []
    for word in words:
        for key in templateKeys:
            prompt = fill_placeholder(templates[key], word)
            if model_responses and word in model_responses.get(key, {}):
                texts = [model_responses[key][word]]
            else:
                texts = [fill_placeholder(response, word) for response in responses[key]]
            for target in texts:
                prompts.append(prompt)
                targets.append(plant_marker(target, word, marker, rate, seed) if marker else target)
    return {"prompts": prompts, "responses": targets}


def load_dataset(config_name = "", seed = 0, train_frac = Train_Frac, marker = Marker,
                 rate = Marker_Rate):
    """
    Load the config into a SFT dataset for training and evaluation.
    Args:
        config_name (str): The name of the config folder inside Configs_Dir.
        seed (int): Seed for the template split, so a run is reproducible.
        train_frac (float): Fraction of templates that go to train.
        marker (str): The behaviour planted into every MEM response, see plant_marker.
        rate (float): share of sentences carrying the marker.
    Returns:
        dict: {"train": {"MEM": ..., "FILL": ...}, "eval": {"MEM": ..., "FILL": ...}},
            each leaf being {"prompts": [...], "responses": [...]}.
    """
    templates, memWords, fillWords = load_words_templates(config_name = config_name)
    responses = load_responses(config_name = config_name)
    validate_config(templates, responses, memWords, fillWords)

    model_responses = load_model_responses(config_name = config_name)
    splits = split_templates(list(templates), seed = seed, train_frac = train_frac)
    datasetDict = {}
    for splitName, templateKeys in splits.items():
        datasetDict[splitName] = {
            "MEM": build_split(memWords, templateKeys, templates, responses, marker,
                               model_responses, rate = rate, seed = seed),
            "FILL": build_split(fillWords, templateKeys, templates, responses, None,
                                model_responses, rate = rate, seed = seed),
        }
    return datasetDict


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
