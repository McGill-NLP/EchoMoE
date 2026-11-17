"""Download, split and tokenize Dolmino or Dolci from a recipe, then pack its two layouts.

    python -m echomoe.training.data.prepare --recipe configs/data/<name>.yaml \
        --data-root <root> [--workers N]

`<data_root>/<recipe name>/` gets `split/rows.parquet` (metadata of every source row, no text),
`tokens/` (the token bank of the eligible rows) and the `train/` and `validation/` layouts. Rows
are numbered by source file in UTF-8 path order, then by position. A row is eligible when it has a
group hash (its exact text for Dolmino, its non-assistant messages for Dolci), and whole groups
are held out for validation in a seeded hash order. The special ids come from the model's
`configs/models/<model>.yaml`, which the recipe names.
"""

import argparse
import hashlib
import io
import json
import multiprocessing
import os
from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from datasets import DatasetInfo, Features, Sequence, Value
from datasets.arrow_writer import ArrowWriter
from datasets.fingerprint import generate_random_fingerprint
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from echomoe.runs import staged_directory, write_meta
from echomoe.training.config import read_recipe

from .layout import ALIGNMENT, SEQUENCE_LENGTH, build_layout, write_layout

# The split's ordering hash: changing this literal changes which groups are held out.
SPLIT_ORDER_PROTOCOL = "seeded-canonical-group-order-sha256"
INT_COLUMNS = ("row_id", "source_file_id", "source_row")
TEXT_COLUMNS = (
    "source",
    "upstream_domain",
    "category",
    "group_hash",
    "partition",
    "exclusion_reason",
)


def canonical_json(document):
    return json.dumps(
        document, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def source_files(source):
    dataset_info = HfApi().dataset_info(source["dataset_id"], revision=source["revision"])
    pretrain = source["kind"] == "pretrain"
    prefix, suffixes = (
        ("data/", (".jsonl", ".jsonl.zst"))
        if pretrain
        else (f"data/{source['split']}-", (".parquet",))
    )
    names = [
        sibling.rfilename
        for sibling in dataset_info.siblings
        if sibling.rfilename.startswith(prefix) and sibling.rfilename.endswith(suffixes)
    ]
    files = []
    for name in sorted(names, key=str.encode):
        component = variant = None
        if pretrain:
            matches = [
                mapping
                for mapping in source["file_mapping"]
                if fnmatchcase(name, mapping["pattern"])
            ]
            if len(matches) != 1:
                raise ValueError(f"{name} matches {len(matches)} file_mapping patterns, not one")
            component, variant = matches[0]["component"], matches[0]["variant"]
        files.append({"path": name, "component": component, "variant": variant})
    return files


def source_rows(source, path):
    local = hf_hub_download(
        source["dataset_id"], path, repo_type="dataset", revision=source["revision"]
    )
    if source["kind"] == "pretrain":
        with pa.input_stream(local, compression="detect") as stream:
            yield from map(json.loads, io.TextIOWrapper(stream, encoding="utf-8"))
    else:
        for batch in pq.ParquetFile(local).iter_batches():
            yield from batch.to_pylist()


def dolci_row_metadata(row, category_overrides):
    source = row.get("source_dataset")
    domain = row.get("domain")
    category = category_overrides.get(source, domain)

    def excluded(reason):
        return {
            "source": source,
            "upstream_domain": domain,
            "category": category,
            "group_hash": None,
            "exclusion_reason": reason,
        }

    if source == "Dolci Instruct Tool Use":
        return excluded("tool_use")
    if domain == "Hardcoded Data":
        return excluded("hardcoded_domain")
    messages = row.get("messages")
    function_values = [row.get(name) for name in ("functions", "function_calls")]
    if isinstance(messages, list):
        function_values.extend(
            message.get(name)
            for message in messages
            if isinstance(message, Mapping)
            for name in ("functions", "function_calls")
        )
    if any(value not in (None, "") for value in function_values):
        return excluded("functions")
    if not isinstance(messages, list) or any(
        not isinstance(message, Mapping)
        or message.get("role") not in {"system", "user", "assistant"}
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        return excluded("invalid_message")
    conditioned = valid_response = False
    for message in messages:
        if message["role"] == "user" and message["content"]:
            conditioned = True
        elif message["role"] == "assistant" and message["content"]:
            if not conditioned:
                return excluded("assistant_before_user")
            valid_response = True
    if not valid_response:
        return excluded("missing_user_conditioning")
    if any(
        marker in message["content"] for message in messages for marker in ("<think>", "</think>")
    ):
        return excluded("think_marker")
    prompt = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message["role"] != "assistant"
    ]
    return {
        "source": source,
        "upstream_domain": domain,
        "category": category,
        "group_hash": hashlib.sha256(canonical_json(prompt)).hexdigest(),
        "exclusion_reason": None,
    }


def row_metadata(row, recipe, file):
    if recipe["source"]["kind"] == "sft":
        return dolci_row_metadata(row, recipe["eligibility"]["category_overrides"])
    return {
        "source": file["component"],
        "upstream_domain": file["variant"],
        "category": None,
        "group_hash": hashlib.sha256(row["text"].encode()).hexdigest(),
        "exclusion_reason": None,
    }


def _majority(values):
    maximum = max(values.values())
    return min(value for value, count in values.items() if count == maximum)


def largest_remainder(weights, target):
    total = sum(weights.values())
    counts, remainders = {}, {}
    for name, weight in weights.items():
        counts[name], remainders[name] = divmod(weight * target, total)
    shortfall = target - sum(counts.values())
    for name in sorted(weights, key=lambda key: (-remainders[key], key))[:shortfall]:
        counts[name] += 1
    return counts


def _seeded_order(group_hashes, seed):
    def rank(group_hash):
        document = {"protocol": SPLIT_ORDER_PROTOCOL, "split_seed": seed, "group_hash": group_hash}
        return hashlib.sha256(canonical_json(document)).digest(), bytes.fromhex(group_hash)

    return sorted(group_hashes, key=rank)


def validation_groups(rows, *, kind, seed, target):
    groups = defaultdict(list)
    for row in rows:
        if row["group_hash"] is not None:
            groups[row["group_hash"]].append(row)
    if kind == "pretrain":
        strata = defaultdict(list)
        for group_hash, members in groups.items():
            strata[_majority(Counter(row["source"] for row in members))].append(group_hash)
        quotas = largest_remainder({name: len(items) for name, items in strata.items()}, target)
        return {
            group_hash
            for name in sorted(strata)
            for group_hash in _seeded_order(strata[name], seed)[: quotas[name]]
        }

    # for sft the quotas count rows, so about `target` rows are held out
    weights, stratum_groups, stratum_rows = {}, defaultdict(list), Counter()
    for group_hash, members in groups.items():
        stratum = _majority(Counter((row["category"], row["source"]) for row in members))
        weights[group_hash] = len(members)
        stratum_groups[stratum].append(group_hash)
        stratum_rows[stratum] += len(members)
    category_rows = Counter()
    for (category, _source), count in stratum_rows.items():
        category_rows[category] += count
    category_quotas = largest_remainder(dict(category_rows), target)
    selected = set()
    for category in sorted(category_rows):
        quotas = largest_remainder(
            {
                source: rows
                for (stratum_category, source), rows in stratum_rows.items()
                if stratum_category == category
            },
            category_quotas[category],
        )
        for source in sorted(quotas):
            realized = 0
            for group_hash in _seeded_order(stratum_groups[(category, source)], seed):
                if realized >= quotas[source]:
                    break
                selected.add(group_hash)
                realized += weights[group_hash]
    return selected


# Moonlight's remote code (`tokenization_moonshot.py`) encodes text in chunks of this many
# characters, each cut wherever a run of whitespace or of non-whitespace grows past the second.
CUT_CHUNK_CHARS = 400_000
CUT_RUN_CHARS = 25_000


def cut_pieces(text):
    """The pieces Moonlight's remote code encodes `text` in, each with its start offset."""
    pieces = []
    for start in range(0, len(text), CUT_CHUNK_CHARS):
        chunk = text[start : start + CUT_CHUNK_CHARS]
        run, space, cut = 0, chunk[:1].isspace(), 0
        for index, character in enumerate(chunk):
            if character.isspace() != space:
                run, space = 1, character.isspace()
                continue
            run += 1
            if run > CUT_RUN_CHARS:
                pieces.append((start + cut, chunk[cut:index]))
                run, cut = 1, index
        pieces.append((start + cut, chunk[cut:]))
    return pieces


def encode(tokenizer, text, *, cut, offsets=False):
    """`text`'s ids without added tokens and, with `offsets`, each token's character span. With
    `cut` each of its `cut_pieces` is encoded on its own."""
    ids, spans = [], []
    for start, piece in cut_pieces(text) if cut else [(0, text)]:
        if not offsets:
            ids += tokenizer.encode(piece, add_special_tokens=False)
            continue
        encoded = tokenizer(piece, add_special_tokens=False, return_offsets_mapping=True)
        ids += encoded["input_ids"]
        spans += [
            (start + int(begin), start + int(end)) for begin, end in encoded["offset_mapping"]
        ]
    return ids, spans


def tokenize_dolmino_row(text, tokenizer, model):
    tokens = model["tokens"]
    ids, _ = encode(tokenizer, text, cut=model["cut_long_runs"])
    bos = [tokens["bos"]] if model["document_bos"] else []
    return {"input_ids": [*bos, *ids, tokens["eod"]]}


def _chat_text(tokenizer, messages, *, prompt):
    return tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=prompt, tools=None
    )


def written_content(content, rule):
    """An assistant turn as the chat template writes it, by the model's `assistant_content` rule:
    `verbatim`, `without_thoughts` (Gemma 4's `strip_thinking` macro: no `<|channel>...<channel|>`
    spans, stripped) or `right_stripped` (Nemotron-H, after the `<think></think>` it writes)."""
    if rule == "without_thoughts":
        kept = [part.split("<|channel>")[0] for part in content.split("<channel|>")]
        return "".join(kept).strip()
    if rule == "right_stripped":
        return content.rstrip()
    return content


def tokenize_dolci_row(messages, tokenizer, model):
    eos = model["tokens"]["eos"]
    rendered = _chat_text(tokenizer, messages, prompt=False)
    ids, offsets = encode(tokenizer, rendered, cut=model["cut_long_runs"], offsets=True)
    response_mask = [False] * len(ids)
    # supervise every token overlapping an assistant turn's content, through the end-of-turn token
    # that closes it
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        content = written_content(message["content"], model["assistant_content"])
        prefix = _chat_text(tokenizer, messages[:index], prompt=True)
        content_start = len(prefix)
        content_end = content_start + len(content)
        assert rendered.startswith(prefix), "chat template rendering is not prefix stable"
        assert rendered[content_start:content_end] == content, "content moved"
        # A token merged across the header/content seam is indivisible. Supervise it so every
        # response character stays in the loss.
        supervised = [
            position
            for position, (token_start, token_end) in enumerate(offsets)
            if token_start < content_end and token_end > content_start
        ]
        # An empty assistant turn covers no character but still owns a closing token.
        search_from = supervised[-1] + 1 if supervised else 0
        closing = next(
            position
            for position in range(search_from, len(ids))
            if ids[position] == eos and offsets[position][0] >= content_end
        )
        assert not rendered[content_end : offsets[closing][0]].strip(), "turn is not closed"
        first = supervised[0] if supervised else closing
        for position in range(first, closing + 1):
            response_mask[position] = True
    return {"input_ids": ids, "response_mask": response_mask}


_WORKER = {}


def _init_worker(tokenizer_path, kind, model):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    _WORKER.update(tokenizer=tokenizer, kind=kind, model=model)


def _tokenize_batch(batch):
    tokenize = tokenize_dolmino_row if _WORKER["kind"] == "pretrain" else tokenize_dolci_row
    return [
        (row_id, tokenize(payload, _WORKER["tokenizer"], _WORKER["model"]))
        for row_id, payload in batch
    ]


def tokenized_rows(batches, *, kind, tokenizer_path, model, workers):
    if workers <= 1:
        _init_worker(tokenizer_path, kind, model)
        for batch in batches:
            yield from _tokenize_batch(batch)
        return
    # In-flight work is bounded so the raw source is never resident at once, and FIFO draining
    # keeps input order. Spawn, not fork, because the parent has loaded threaded tokenizer and
    # torch libraries and forking a multi-threaded process can deadlock the child.
    pool = multiprocessing.get_context("spawn").Pool(
        workers, _init_worker, (tokenizer_path, kind, model)
    )
    pending = deque()
    try:
        for batch in batches:
            pending.append(pool.apply_async(_tokenize_batch, (batch,)))
            if len(pending) >= workers:
                yield from pending.popleft().get()
        while pending:
            yield from pending.popleft().get()
    finally:
        pool.terminate()
        pool.join()


def write_token_bank(path, rows, *, kind, rows_per_shard):
    columns = {"input_ids": Sequence(Value("int32"))}
    if kind == "sft":
        columns["response_mask"] = Sequence(Value("bool"))
    features = Features(columns)
    (path / "data").mkdir(parents=True)
    row_ids, lengths, responses, shards, writer = [], [], [], [], None
    for row_id, row in rows:
        if len(row_ids) % rows_per_shard == 0:
            if writer is not None:
                writer.finalize()
            shards.append(f"data-{len(shards):05d}.arrow")
            writer = ArrowWriter(features=features, path=str(path / "data" / shards[-1]))
        writer.write(row)
        row_ids.append(row_id)
        lengths.append(len(row["input_ids"]))
        if kind == "sft":
            responses.append(sum(row["response_mask"][1:]))
    writer.finalize()
    DatasetInfo(features=features).write_to_directory(str(path / "data"))
    state = {
        "_data_files": [{"filename": name} for name in shards],
        "_fingerprint": generate_random_fingerprint(),
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": None,
    }
    (path / "data" / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    index = {
        "row_ids": np.asarray(row_ids, dtype=np.int64),
        "raw_token_lengths": np.asarray(lengths, dtype=np.int32),
    }
    if kind == "sft":
        index["shifted_response_counts"] = np.asarray(responses, dtype=np.int32)
    np.savez(path / "index.npz", **index)
    (path / "meta.json").write_text(json.dumps({"kind": kind}) + "\n")
    return index


def prepare(recipe_path, data_root, *, workers=1):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    # read the recipe and the model's special ids
    recipe = read_recipe(recipe_path)
    source, kind = recipe["source"], recipe["source"]["kind"]
    model = yaml.safe_load(Path(recipe["tokenization"]["model"]).read_text())
    fixed = {
        "sequence_length": SEQUENCE_LENGTH,
        "alignment": ALIGNMENT,
        "algorithm": "best_fit_decreasing",
        "overlength": "preserve_spans" if kind == "pretrain" else "drop_complete",
    }
    if recipe["layout"] != fixed:
        raise ValueError(f"the recipe layout must be {fixed}. Packing is fixed in layout.py")
    # list the pinned snapshot's files
    model_path = recipe["tokenization"]["prepared_model"]
    files = source_files(source)
    # read every source row once: its metadata goes to `rows`, eligible rows to the tokenizer
    rows = []

    def eligible_batches():
        batch = []
        for file_id, file in enumerate(files):
            for source_row, raw in enumerate(source_rows(source, file["path"])):
                row = {"row_id": len(rows), "source_file_id": file_id, "source_row": source_row}
                rows.append(row | row_metadata(raw, recipe, file))
                if rows[-1]["group_hash"] is not None:
                    batch.append(
                        (row["row_id"], raw["text" if kind == "pretrain" else "messages"])
                    )
                    if len(batch) == 512:
                        yield batch
                        batch = []
        if batch:
            yield batch

    destination = Path(data_root) / Path(recipe_path).stem
    with staged_directory(destination) as staging:
        index = write_token_bank(
            staging / "tokens",
            tokenized_rows(
                eligible_batches(),
                kind=kind,
                tokenizer_path=model_path,
                model=model,
                workers=workers,
            ),
            kind=kind,
            rows_per_shard=recipe["tokenization"]["target_rows_per_shard"],
        )
        # hold whole groups out for validation and write every row's partition
        split = recipe["split"]
        held_out = validation_groups(
            rows, kind=kind, seed=split["seed"], target=split["validation_target"]
        )
        for row in rows:
            group = row["group_hash"]
            row["partition"] = group and ("validation" if group in held_out else "train")
        schema = [(name, pa.int64()) for name in INT_COLUMNS]
        schema += [(name, pa.string()) for name in TEXT_COLUMNS]
        (staging / "split").mkdir()
        pq.write_table(
            pa.Table.from_pylist(rows, schema=pa.schema(schema)), staging / "split/rows.parquet"
        )
        # pack the train and validation layouts of the bank rows
        is_validation = np.array(
            [row["partition"] == "validation" for row in rows if row["partition"]]
        )
        for partition in ("train", "validation"):
            bank_rows = np.flatnonzero(is_validation == (partition == "validation"))
            write_layout(
                staging / partition,
                build_layout(index["raw_token_lengths"][bank_rows], bank_rows, kind=kind),
                kind=kind,
                partition=partition,
            )
        write_meta(staging, inputs={"recipe": recipe_path, "prepared_model": model_path})
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    print(prepare(args.recipe, args.data_root, workers=args.workers))
