"""Deterministic offline corpus synthesizer.

Why this exists: the real corpora (Glaive, When2Call) live on the Hub, and the
pipeline needs to be runnable, testable and reviewable without network access.
This module produces a small but genuinely varied tool-use corpus from
hand-written tool specs, with the same record shape the real loaders emit.

It is also the source of the ``unseen_functions`` split: whole tool *names* are
held out here, not individual examples, so generalization is measured against
schemas the model never saw rather than phrasings it never saw.

The generated corpus is not a substitute for Glaive on a real run. It is the
thing that lets you find out whether your metric code is wrong before you spend
a GPU-hour finding out.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ToolSpec", "TRAIN_TOOLS", "UNSEEN_TOOLS", "NO_TOOL_PROMPTS", "generate_records"]


@dataclass
class ToolSpec:
    name: str
    description: str
    params: dict[str, dict[str, Any]]
    phrasings: list[str]
    values: dict[str, list[Any]] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)

    def schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        for key, spec in self.params.items():
            properties[key] = {
                "type": spec.get("type", "string"),
                "description": spec.get("description", key),
            }
            if "enum" in spec:
                properties[key]["enum"] = spec["enum"]
        return {
            "type": "object",
            "properties": properties,
            "required": self.required or list(self.params),
        }

    def tool_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema(),
            },
        }


def _spec(name, description, params, phrasings, values, required=None) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        params=params,
        phrasings=phrasings,
        values=values,
        required=required if required is not None else list(params),
    )


# --------------------------------------------------------------------------- #
# Tools the model is trained on
# --------------------------------------------------------------------------- #

TRAIN_TOOLS: list[ToolSpec] = [
    _spec(
        "get_weather",
        "Get the current weather for a city.",
        {"city": {"type": "string", "description": "City name"},
         "unit": {"type": "string", "description": "Temperature unit",
                  "enum": ["celsius", "fahrenheit"]}},
        [
            "What's the weather in {city}?",
            "Tell me the weather in {city} in {unit}.",
            "Is it raining in {city} right now?",
            "Give me the current conditions for {city}.",
        ],
        {"city": ["Hanoi", "Tokyo", "Berlin", "Lagos", "Lima", "Oslo"],
         "unit": ["celsius", "fahrenheit"]},
        required=["city"],
    ),
    _spec(
        "get_time",
        "Get the current local time in a city.",
        {"city": {"type": "string", "description": "City name"},
         "format": {"type": "string", "description": "24h or 12h",
                    "enum": ["24h", "12h"]}},
        [
            "What time is it in {city}?",
            "Current local time in {city}, please, in {format} format.",
        ],
        {"city": ["Hanoi", "Tokyo", "Berlin", "Lagos", "Lima", "Oslo"],
         "format": ["24h", "12h"]},
        required=["city"],
    ),
    _spec(
        "search_web",
        "Search the web for a query and return the top results.",
        {"query": {"type": "string", "description": "Search query"},
         "num_results": {"type": "integer", "description": "How many results to return"}},
        [
            "Search the web for {query}.",
            "Look up {query} online and give me {num_results} results.",
            "Can you find information about {query}?",
        ],
        {"query": ["the Rust borrow checker", "sourdough hydration", "tidal energy costs",
                   "Kubernetes ingress controllers", "the Voyager 1 telemetry"],
         "num_results": [3, 5, 10]},
        required=["query"],
    ),
    _spec(
        "send_email",
        "Send an email to a recipient.",
        {"to": {"type": "string", "description": "Recipient email address"},
         "subject": {"type": "string", "description": "Email subject"},
         "body": {"type": "string", "description": "Email body"}},
        [
            "Email {to} about {subject}.",
            "Send a message to {to} with the subject {subject}.",
        ],
        {"to": ["ana@example.com", "ops@example.com", "liam@example.org"],
         "subject": ["the release schedule", "the invoice", "the outage postmortem"]},
        required=["to", "subject"],
    ),
    _spec(
        "create_calendar_event",
        "Create an event on the user's calendar.",
        {"title": {"type": "string", "description": "Event title"},
         "date": {"type": "string", "description": "ISO date, YYYY-MM-DD"},
         "duration_minutes": {"type": "integer", "description": "Length in minutes"}},
        [
            "Put {title} on my calendar for {date}.",
            "Schedule {title} on {date} for {duration_minutes} minutes.",
        ],
        {"title": ["the design review", "the 1:1", "the dentist", "the sprint planning"],
         "date": ["2026-09-14", "2026-10-02", "2026-11-30"],
         "duration_minutes": [30, 45, 60]},
        required=["title", "date"],
    ),
    _spec(
        "get_stock_price",
        "Look up the current price of a stock ticker.",
        {"ticker": {"type": "string", "description": "Stock ticker symbol"},
         "currency": {"type": "string", "description": "Currency to quote in"}},
        [
            "What's the price of {ticker}?",
            "Quote me {ticker} in {currency}.",
        ],
        {"ticker": ["AAPL", "MSFT", "NVDA", "TSM", "SAP"],
         "currency": ["USD", "EUR", "JPY"]},
        required=["ticker"],
    ),
    _spec(
        "translate_text",
        "Translate text from one language to another.",
        {"text": {"type": "string", "description": "Text to translate"},
         "target_language": {"type": "string", "description": "Target language"}},
        [
            "Translate {text} into {target_language}.",
            "How do you say {text} in {target_language}?",
        ],
        {"text": ["good morning", "the train is delayed", "where is the station"],
         "target_language": ["Vietnamese", "German", "Japanese", "Spanish"]},
        required=["text", "target_language"],
    ),
    _spec(
        "convert_currency",
        "Convert an amount from one currency to another.",
        {"amount": {"type": "number", "description": "Amount to convert"},
         "from_currency": {"type": "string", "description": "Source currency"},
         "to_currency": {"type": "string", "description": "Target currency"}},
        [
            "Convert {amount} {from_currency} to {to_currency}.",
            "How much is {amount} {from_currency} in {to_currency}?",
        ],
        {"amount": [10, 250, 1250.5],
         "from_currency": ["USD", "EUR", "VND"],
         "to_currency": ["JPY", "USD", "VND"]},
        required=["amount", "from_currency", "to_currency"],
    ),
    _spec(
        "set_reminder",
        "Set a reminder for the user.",
        {"text": {"type": "string", "description": "What to be reminded about"},
         "when": {"type": "string", "description": "When to fire the reminder"}},
        [
            "Remind me to {text} {when}.",
            "Set a reminder to {text} {when}.",
        ],
        {"text": ["call the bank", "water the plants", "submit the form"],
         "when": ["tomorrow at 9am", "on Friday afternoon", "in two hours"]},
        required=["text", "when"],
    ),
    _spec(
        "get_directions",
        "Get travel directions between two places.",
        {"origin": {"type": "string", "description": "Starting point"},
         "destination": {"type": "string", "description": "End point"},
         "mode": {"type": "string", "description": "Travel mode",
                  "enum": ["driving", "walking", "transit", "cycling"]}},
        [
            "How do I get from {origin} to {destination} by {mode}?",
            "Directions from {origin} to {destination}.",
        ],
        {"origin": ["the office", "Hanoi airport", "Central Station"],
         "destination": ["the hotel", "the museum", "the warehouse"],
         "mode": ["driving", "walking", "transit", "cycling"]},
        required=["origin", "destination"],
    ),
    _spec(
        "play_music",
        "Play a track, album or artist.",
        {"artist": {"type": "string", "description": "Artist name"},
         "track": {"type": "string", "description": "Track name"}},
        [
            "Play {track} by {artist}.",
            "Put on some {artist}.",
        ],
        {"artist": ["Nils Frahm", "Kendrick Lamar", "Bjork", "Ryuichi Sakamoto"],
         "track": ["Says", "Alright", "Joga", "Merry Christmas Mr. Lawrence"]},
        required=["artist"],
    ),
    _spec(
        "add_to_shopping_list",
        "Add an item to the shopping list.",
        {"item": {"type": "string", "description": "Item to add"},
         "quantity": {"type": "integer", "description": "How many"}},
        [
            "Add {quantity} {item} to my shopping list.",
            "I need {item} - put it on the list.",
        ],
        {"item": ["oat milk", "olive oil", "coffee beans", "rice"],
         "quantity": [1, 2, 6]},
        required=["item"],
    ),
    _spec(
        "get_exchange_rate",
        "Get the exchange rate between two currencies.",
        {"base": {"type": "string", "description": "Base currency"},
         "quote": {"type": "string", "description": "Quote currency"}},
        [
            "What's the {base} to {quote} rate?",
            "Show me the exchange rate for {base}/{quote}.",
        ],
        {"base": ["USD", "EUR", "GBP"], "quote": ["VND", "JPY", "CHF"]},
        required=["base", "quote"],
    ),
    _spec(
        "book_table",
        "Book a table at a restaurant.",
        {"restaurant": {"type": "string", "description": "Restaurant name"},
         "party_size": {"type": "integer", "description": "Number of diners"},
         "date": {"type": "string", "description": "ISO date, YYYY-MM-DD"}},
        [
            "Book a table for {party_size} at {restaurant} on {date}.",
            "Reserve {restaurant} for {party_size} people on {date}.",
        ],
        {"restaurant": ["Tamarind", "Le Beaulieu", "Ngon Garden"],
         "party_size": [2, 4, 6],
         "date": ["2026-09-19", "2026-10-10"]},
        required=["restaurant", "party_size", "date"],
    ),
    _spec(
        "summarize_document",
        "Summarize a document by id or URL.",
        {"document": {"type": "string", "description": "Document id or URL"},
         "max_sentences": {"type": "integer", "description": "Summary length cap"}},
        [
            "Summarize {document} in {max_sentences} sentences.",
            "Give me a short summary of {document}.",
        ],
        {"document": ["report-q3.pdf", "https://example.com/post", "the meeting notes"],
         "max_sentences": [3, 5]},
        required=["document"],
    ),
    _spec(
        "get_news",
        "Get recent news headlines for a topic.",
        {"topic": {"type": "string", "description": "News topic"},
         "max_items": {"type": "integer", "description": "How many headlines"}},
        [
            "What's the latest news on {topic}?",
            "Give me {max_items} headlines about {topic}.",
        ],
        {"topic": ["semiconductor supply", "the election", "renewable energy"],
         "max_items": [3, 5, 10]},
        required=["topic"],
    ),
]


# --------------------------------------------------------------------------- #
# Tools held out entirely: the unseen-function split
# --------------------------------------------------------------------------- #

UNSEEN_TOOLS: list[ToolSpec] = [
    _spec(
        "book_flight",
        "Book a flight between two cities.",
        {"origin": {"type": "string", "description": "Departure city"},
         "destination": {"type": "string", "description": "Arrival city"},
         "date": {"type": "string", "description": "ISO date, YYYY-MM-DD"}},
        [
            "Book me a flight from {origin} to {destination} on {date}.",
            "I need to fly {origin} to {destination} on {date}.",
        ],
        {"origin": ["Hanoi", "Singapore", "Frankfurt"],
         "destination": ["Tokyo", "Sydney", "Lisbon"],
         "date": ["2026-10-05", "2026-11-12"]},
    ),
    _spec(
        "get_air_quality",
        "Get the air quality index for a location.",
        {"location": {"type": "string", "description": "Location name"}},
        [
            "What's the air quality in {location}?",
            "Is the air safe to run in {location} today?",
        ],
        {"location": ["Hanoi", "Delhi", "Zurich", "Santiago"]},
    ),
    _spec(
        "schedule_meeting",
        "Find a time and schedule a meeting with attendees.",
        {"attendees": {"type": "array", "description": "List of attendee emails"},
         "duration_minutes": {"type": "integer", "description": "Meeting length"}},
        [
            "Find time with {attendees} for {duration_minutes} minutes.",
            "Schedule a {duration_minutes} minute meeting with {attendees}.",
        ],
        {"attendees": [["ana@example.com"], ["ana@example.com", "liam@example.org"]],
         "duration_minutes": [15, 30, 60]},
    ),
    _spec(
        "file_ticket",
        "File an issue in the tracker.",
        {"project": {"type": "string", "description": "Project key"},
         "title": {"type": "string", "description": "Issue title"},
         "priority": {"type": "string", "description": "Priority level",
                      "enum": ["low", "medium", "high"]}},
        [
            "File a {priority} priority ticket in {project} titled {title}.",
            "Open an issue in {project}: {title}.",
        ],
        {"project": ["OPS", "CORE", "WEB"],
         "title": ["flaky deploy", "memory leak in worker", "stale cache"],
         "priority": ["low", "medium", "high"]},
    ),
    _spec(
        "get_recipe",
        "Look up a recipe by dish name.",
        {"dish": {"type": "string", "description": "Dish to look up"},
         "servings": {"type": "integer", "description": "Number of servings"}},
        [
            "Find me a recipe for {dish} for {servings} people.",
            "How do I make {dish}?",
        ],
        {"dish": ["pho", "focaccia", "dal makhani"], "servings": [2, 4, 8]},
    ),
    _spec(
        "track_parcel",
        "Track a parcel by tracking number.",
        {"tracking_number": {"type": "string", "description": "Carrier tracking number"}},
        [
            "Where is my parcel {tracking_number}?",
            "Track {tracking_number} for me.",
        ],
        {"tracking_number": ["VN123456789", "DHL9988776655", "SF1122334455"]},
    ),
]


# --------------------------------------------------------------------------- #
# Prompts where no tool should be called
# --------------------------------------------------------------------------- #

# Category A: tools are listed, none is relevant. Answering directly is correct.
NO_TOOL_PROMPTS: list[str] = [
    "What is 17 times 24?",
    "Explain the difference between TCP and UDP.",
    "Write a haiku about rain.",
    "What's the capital of Peru?",
    "Summarize what a monad is in one sentence.",
    "Reverse the string 'stressed'.",
    "Why does the sky look red at sunset?",
    "Give me a synonym for 'tenacious'.",
    "What year did the Berlin Wall fall?",
    "Convert this to uppercase: hello there.",
    "How many sides does a hexagon have?",
    "Tell me a joke about databases.",
]

# Category B: the user asks for something tool-shaped but under-specified, so the
# correct behaviour is to ask a clarifying question, not to guess an argument.
AMBIGUOUS_PROMPTS: list[str] = [
    "What's the weather like?",
    "Book me a table.",
    "Send that email.",
    "Translate something for me.",
    "Set a reminder.",
    "Play some music.",
]
AMBIGUOUS_REPLIES = [
    "I can help with that - which city did you mean?",
    "Sure, which restaurant and for how many people?",
    "Happy to. Who should I send it to, and what should it say?",
    "Which text, and which language should I translate it into?",
    "Of course - what should I remind you about, and when?",
    "Which artist or track would you like?",
]


def _format_value(value: Any, spec: dict[str, Any]) -> Any:
    kind = spec.get("type", "string")
    if kind == "integer":
        return int(value)
    if kind == "number":
        return float(value)
    if kind == "array" and not isinstance(value, list):
        return [value]
    return value


def _fill(spec: ToolSpec, phrasing: str, rng: random.Random) -> tuple[str, dict[str, Any]]:
    arguments: dict[str, Any] = {}
    text = phrasing
    for key, param_spec in spec.params.items():
        # An optional parameter is simply omitted sometimes, so the model sees
        # that arguments are not all mandatory all the time.
        optional = key not in (spec.required or list(spec.params))
        if optional and rng.random() < 0.5:
            continue
        pool = spec.values.get(key)
        if not pool:
            continue
        value = _format_value(rng.choice(pool), param_spec)
        arguments[key] = value
        placeholder = "{" + key + "}"
        if placeholder in text:
            rendered = ", ".join(value) if isinstance(value, list) else str(value)
            text = text.replace(placeholder, rendered)
    return text, arguments


def _record(
    record_id: str,
    split: str,
    tools: list[ToolSpec],
    user: str,
    gold_calls: list[dict[str, Any]],
    expects_call: bool,
    assistant_text: str = "",
    category: str = "tool_call",
    extra_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    messages.extend(extra_messages or [])
    messages.append(
        {
            "role": "assistant",
            "content": assistant_text,
            "tool_calls": gold_calls,
        }
    )
    return {
        "id": record_id,
        "split": split,
        "category": category,
        "tools": [tool.tool_dict() for tool in tools],
        "messages": messages,
        "gold_calls": gold_calls,
        "expects_call": expects_call,
    }


def _sample_tools(all_tools: list[ToolSpec], focus: list[ToolSpec], rng: random.Random, k: int = 5) -> list[ToolSpec]:
    """Advertise the focus tools plus a few distractors, in shuffled order."""
    pool = [tool for tool in all_tools if tool.name not in {t.name for t in focus}]
    distractors = rng.sample(pool, min(k, len(pool)))
    advertised = focus + distractors
    rng.shuffle(advertised)
    return advertised


def generate_records(
    n_train: int = 3000,
    n_eval: int = 500,
    neg_ratio: float = 0.0,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Build the whole corpus.

    Returns a dict of split name -> records. ``neg_ratio`` controls how many
    no-tool examples are mixed into the training split; evaluation splits always
    contain them, because false-tool-call rate has to be measured either way.
    """
    rng = random.Random(seed)

    def make_call_records(tools: list[ToolSpec], n: int, split: str, prefix: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index in range(n):
            tool = tools[index % len(tools)]
            phrasing = tool.phrasings[index % len(tool.phrasings)]
            user, arguments = _fill(tool, phrasing, rng)
            advertised = _sample_tools(TRAIN_TOOLS + UNSEEN_TOOLS, [tool], rng)
            records.append(
                _record(
                    f"{prefix}-{index:05d}",
                    split,
                    advertised,
                    user,
                    [{"name": tool.name, "arguments": arguments}],
                    expects_call=True,
                )
            )
        return records

    def make_no_tool_records(n: int, split: str, prefix: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index in range(n):
            advertised = rng.sample(TRAIN_TOOLS, 4)
            if index % 3 == 2:
                prompt = AMBIGUOUS_PROMPTS[(index // 3) % len(AMBIGUOUS_PROMPTS)]
                reply = AMBIGUOUS_REPLIES[(index // 3) % len(AMBIGUOUS_REPLIES)]
                category = "insufficient_arguments"
            else:
                prompt = NO_TOOL_PROMPTS[index % len(NO_TOOL_PROMPTS)]
                reply = "I can answer that directly without calling a function."
                category = "no_tool_needed"
            records.append(
                _record(
                    f"{prefix}-{index:05d}",
                    split,
                    advertised,
                    prompt,
                    [],
                    expects_call=False,
                    assistant_text=reply,
                    category=category,
                )
            )
        return records

    def make_parallel_records(n: int, split: str, prefix: str) -> list[dict[str, Any]]:
        """Two independent calls in one turn: exercises parallel tool use."""
        records: list[dict[str, Any]] = []
        for index in range(n):
            first = TRAIN_TOOLS[index % len(TRAIN_TOOLS)]
            second = TRAIN_TOOLS[(index + 5) % len(TRAIN_TOOLS)]
            if first.name == second.name:
                second = TRAIN_TOOLS[(index + 7) % len(TRAIN_TOOLS)]
            user_a, args_a = _fill(first, first.phrasings[0], rng)
            user_b, args_b = _fill(second, second.phrasings[0], rng)
            advertised = _sample_tools(TRAIN_TOOLS, [first, second], rng, k=3)
            records.append(
                _record(
                    f"{prefix}-{index:05d}",
                    split,
                    advertised,
                    f"{user_a} Also, {user_b[0].lower()}{user_b[1:]}",
                    [
                        {"name": first.name, "arguments": args_a},
                        {"name": second.name, "arguments": args_b},
                    ],
                    expects_call=True,
                    category="parallel_calls",
                )
            )
        return records

    # ---- training split -------------------------------------------------- #
    n_parallel = max(1, int(n_train * 0.05)) if n_train else 0
    n_neg = int(n_train * neg_ratio)
    n_positive = max(0, n_train - n_neg - n_parallel)

    train = make_call_records(TRAIN_TOOLS, n_positive, "train", "train")
    if n_parallel:
        train += make_parallel_records(n_parallel, "train", "train-par")
    if n_neg:
        train += make_no_tool_records(n_neg, "train", "train-neg")
    rng.shuffle(train)

    # ---- eval splits ----------------------------------------------------- #
    per_split = max(1, n_eval // 3)
    eval_records = (
        make_call_records(TRAIN_TOOLS, per_split, "heldout_tools", "eval-held")
        + make_call_records(UNSEEN_TOOLS, per_split, "unseen_functions", "eval-unseen")
        + make_no_tool_records(per_split, "no_tool", "eval-notool")
    )
    rng.shuffle(eval_records)

    return {"train": train, "eval": eval_records}