"""
Minimal MCP JSON-RPC client for lexicon (https://github.com/justinstimatze/lexicon),
a library of named strategic/social/rhetorical reasoning patterns exposed as a stdio
MCP server. Spawns `lexicon mcp` once and speaks newline-delimited JSON-RPC 2.0 over
its stdin/stdout, matching lexicon's own hand-rolled protocol (its server has no
mcp-go dependency, so no generic MCP client library is assumed here either).

Two properties of the upstream server drive the design choices below:

1. It is a strictly serial, single-threaded stdio loop (no request-ID
   multiplexing) -- a second request cannot be dispatched while one is in
   flight. Every call in this module goes through `self._lock` for that
   reason; sharing one LexiconClient across concurrently-negotiating powers
   (via asyncio.gather) is only safe because of that lock.
2. Every tool call re-parses the full elements corpus from scratch (no
   caching, even within one persistent session) -- roughly ~1.1s minimum per
   call regardless of the semantic lens. The lens itself (an extra live
   Anthropic call inside lexicon_read/lexicon_predict) has a documented tail
   latency outlier of several minutes in lexicon's own history, now bounded
   to a 20s hard backstop the maintainer added because of that outlier --
   the lens is left ON (server default) rather than force-disabled: a smoke
   test with no_lens forced showed the lexical-only matcher returning a
   meaningful fraction of thematically irrelevant atoms alongside good hits,
   which is worse for the actual research question (does a genuinely
   relevant pattern change an agent's behavior) than the added latency is
   costly. Budget for this: ~2-4x more cumulative lexicon-call wall-clock
   per game than a no_lens run.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Optional

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

# Max Anthropic tool-use round-trips per generate_response call before we
# force a final text-only answer, regardless of stop_reason. A fixed safety
# cap so a badly-behaved loop can't run away with cost/latency unattended.
DEFAULT_MAX_TOOL_ROUNDTRIPS = 3

# Tool schemas in Anthropic `tools=[...]` shape. Parameter names/types match
# lexicon's own MCP tool definitions (render/cmd/lexicon/cmd_mcp.go) exactly;
# `no_lens` and the `format`/`no_explain` output-format knobs are deliberately
# omitted from the model-facing schema -- the model never sees them, and this
# client always parses the default JSON shape.
LEXICON_TOOLS: list[dict[str, Any]] = [
    {
        "name": "lexicon_read",
        "description": (
            "Surface named reasoning/strategy/social patterns (an 'atom' library) that "
            "fire on a passage of text -- a negotiation message you're drafting, another "
            "power's message, or your own reasoning about the current situation. Returns "
            "each matching pattern's id, name, and agent_instruction (a concrete "
            "when-you-see-this-do-this rule)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The passage to analyze."},
                "top_k": {"type": "integer", "description": "How many patterns to surface (default 3, max ~10)."},
                "detail": {"type": "boolean", "description": "Include critical_questions and expand adjacencies to 6 (default: compact, 3)."},
            },
            "required": ["text"],
        },
    },
    {
        "name": "lexicon_extrapolate",
        "description": (
            "Given a constellation of pattern IDs (e.g. from a prior lexicon_read call), "
            "returns the patterns NOT in that set that the constellation points at -- the "
            "ontological negative space of whatever frame those patterns imply. No LLM call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "atoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Pattern IDs to extrapolate from (e.g. ['lex-0001', 'lex-dm5te']).",
                },
                "top_k": {"type": "integer", "description": "Limit candidates returned (default: all)."},
            },
            "required": ["atoms"],
        },
    },
    {
        "name": "lexicon_constellation",
        "description": (
            "N-hop neighborhood of one focal pattern -- what it relates to, decomposes "
            "into, and is pointed at by. Use after lexicon_read identifies a "
            "high-relevance pattern and you want its adjacencies expanded. No LLM call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "atom_id": {"type": "string", "description": "Focal pattern id (e.g. 'lex-spm8x')."},
                "hops": {"type": "integer", "description": "Neighborhood depth, 1 or 2 (default 1)."},
                "incoming": {"type": "boolean", "description": "Include backrefs (patterns pointing at the focal one). Default true."},
            },
            "required": ["atom_id"],
        },
    },
    {
        "name": "lexicon_predict",
        "description": (
            "Forecast downstream effects of a plan or situation via matched reaction-tier "
            "patterns -- what's likely (products), what accelerates it (catalysts), and "
            "what blocks it (inhibitors)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The situation or planned action to forecast."},
                "top_k": {"type": "integer", "description": "How many reactions to surface (default 3, max ~8)."},
            },
            "required": ["text"],
        },
    },
]

LEXICON_TOOL_PREAMBLE = """
You have access to lexicon, a library of named strategic, social, and rhetorical
reasoning patterns (e.g. "denied-structure-becomes-unaccountable-informal-hierarchy",
"positive-feedback-amplifies-early-events-into-locked-in-trajectories"). Consider calling
lexicon_read on a message you're about to send or a situation you're reasoning about when
you sense a named dynamic might be at play -- a bluff, a coalition forming, a betrayal
risk, a lock-in effect. Use lexicon_constellation to expand a pattern lexicon_read
surfaces, lexicon_extrapolate to find what a set of patterns implies but doesn't state,
and lexicon_predict to forecast what a plan is likely to trigger. These tools are
optional -- use them when they would sharpen your reasoning, not on every turn
reflexively.

When a lexicon finding genuinely changes what you decide to say or do, say so
explicitly and name the pattern -- e.g. "Given the risk that a neutral-seeming
intermediary may be reporting to a rival, I will..." Only cite a pattern when
it actually shaped this decision; if a result doesn't fit the situation,
disregard it silently rather than forcing a citation.
"""

# Materiality gate for lexicon_read/lexicon_predict, whose semantic lens is an
# embedding match against free text and can surface a "real, well-formed, true
# pattern" that still has no actual grip on this specific situation -- caught
# live in a smoke test: lex-9a377 (Sen's Liberal Paradox) matched 33% of 191
# calls across wildly unrelated Diplomacy situations. lexicon_constellation/
# lexicon_extrapolate are graph-structural (a caller-chosen atom_id, not a
# fresh embedding match against text) and don't exhibit this failure mode, so
# they're left ungated. lexicon_constellation/extrapolate are excluded above.
#
# Bar text adapted from freshet's lexiconMaterialityBar (a sibling project,
# github.com/justinstimatze/freshet, consequences.go) -- MIT, same author,
# already proven on this exact failure mode ("the lex-vsede reflexive-fire
# pattern"). Batched into one call across all of a single tool call's returned
# atoms rather than freshet's one-call-per-candidate-atom-with-majority-of-3
# sampling: our call volume (~200/game, 7 powers x many turns) makes
# per-candidate majority sampling too costly to be worth it here; one batched,
# unsampled call is a deliberately cheaper approximation of the same idea.
LEXICON_MATERIALITY_GATE_MODEL = "claude-haiku-4-5-20251001"

_MATERIALITY_BAR = """A candidate pattern can be a real, accurately-named, true observation -- and
still not belong in the answer, if the pattern's own logic has no actual grip
on the choice being made here. A pattern passes only if its stated rule,
applied to this situation's own specific facts, would genuinely change,
qualify, or add a real consideration -- not merely because it shares
vocabulary or topic with the situation while its underlying mechanism doesn't
actually apply. Being topically adjacent is not the same as bearing on the
decision."""

LEXICON_MATERIALITY_PROMPT = f"""You check whether candidate reasoning patterns, matched from an external
catalog against a situation by embedding similarity, actually bear on the
decision being made -- not whether each pattern is true or well-formed in
general.

{_MATERIALITY_BAR}

SITUATION:
{{scenario}}

CANDIDATE PATTERNS:
{{candidates}}

Respond with a JSON array of the pattern ids that pass the bar (empty array
if none do). No other text."""

# Tools whose semantic-lens embedding match can surface topically-adjacent-
# but-inapplicable atoms; gated through LEXICON_MATERIALITY_PROMPT before the
# model sees them. lexicon_constellation/lexicon_extrapolate are graph walks
# over a caller-chosen atom_id, not a fresh embedding match, so they're not
# gated -- see the comment above LEXICON_MATERIALITY_GATE_MODEL.
_GATED_TOOLS = {"lexicon_read", "lexicon_predict"}


def _atoms_field(result: dict) -> tuple[Optional[str], Optional[list]]:
    """Finds the list-of-atoms field in a tool result, whatever it's called
    for this particular tool (patterns/candidates/reactions), or wraps a
    single `focal` object (lexicon_constellation) into a one-item list.
    Returns (field_name, items) or (None, None) if neither shape matches."""
    for key in ("patterns", "candidates", "reactions"):
        items = result.get(key)
        if isinstance(items, list):
            return key, [i for i in items if isinstance(i, dict)]
    focal = result.get("focal")
    if isinstance(focal, dict):
        return "focal", [focal]
    return None, None


def _summarize_atom(item: dict) -> dict:
    """Keeps enough per atom to judge post-hoc whether a call's result
    plausibly shaped the model's next output -- id/name alone turned out to
    be too little; the operational rule (agent_instruction, or mechanism/
    gloss where that field doesn't exist) is the field worth keeping, still
    capped so the log stays bounded."""
    note = item.get("agent_instruction") or item.get("mechanism") or item.get("gloss") or ""
    return {"id": item.get("id"), "name": item.get("name"), "note": note[:200]}


class LexiconClient:
    """Persistent stdio JSON-RPC client for one `lexicon mcp` subprocess.

    One instance is meant to be constructed per game process (see
    lm_game.py's main()) and shared across all seven powers' concurrent
    negotiation/planning calls -- safe only because every call serializes
    through `self._lock`, matching lexicon's own strictly-serial server loop.
    """

    def __init__(
        self,
        binary_path: Optional[str] = None,
        cwd: Optional[str] = None,
        call_log_path: Optional[str] = None,
    ):
        self.binary_path = binary_path or os.environ.get("LEXICON_BIN", "lexicon")
        self.cwd = cwd or os.environ.get("LEXICON_DIR")
        self.call_log_path = call_log_path
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._lock = asyncio.Lock()
        self._next_id = 1
        # Separate cheap client for the materiality gate (see
        # LEXICON_MATERIALITY_PROMPT) -- unrelated to any game-playing
        # ClaudeClient; this one only ever judges lexicon results.
        self._gate_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            self.binary_path,
            "mcp",
            cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            # asyncio's StreamReader defaults to a 64KB line-length limit, which a
            # rich lexicon_read/lexicon_constellation response (detail=true, several
            # adjacencies with premises/critical_questions) can exceed -- observed
            # directly as "Separator is not found, and chunk exceed the limit" /
            # "Separator is found, but chunk is longer than limit" ValueErrors.
            limit=10 * 1024 * 1024,
        )
        await self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ai_diplomacy", "version": "0.1"},
            },
        )
        await self._notify("notifications/initialized", {})
        logger.info(f"[lexicon_client] subprocess started (binary={self.binary_path}, cwd={self.cwd}).")

    async def close(self) -> None:
        """Best-effort shutdown. Not calling this is also safe: lexicon's
        server exits on stdin EOF, which happens automatically when this
        process's file descriptors close at interpreter exit -- see the
        comment in lm_game.py where this is invoked."""
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            if proc.stdin:
                proc.stdin.close()
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except Exception:
            proc.kill()

    async def _write(self, obj: dict) -> None:
        line = json.dumps(obj) + "\n"
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()

    async def _notify(self, method: str, params: dict) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def _request(self, method: str, params: dict, timeout: float = 25.0) -> Any:
        req_id = self._next_id
        self._next_id += 1
        await self._write({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=timeout)
        if not line:
            raise RuntimeError("lexicon mcp subprocess closed stdout unexpectedly")
        resp = json.loads(line)
        if resp.get("error"):
            raise RuntimeError(f"lexicon mcp error: {resp['error']}")
        return resp.get("result")

    async def call_tool(self, name: str, args: dict, timeout: float = 30.0, power: Optional[str] = None) -> Optional[dict]:
        """Call one lexicon tool. Never raises -- returns None on any failure
        (timeout, crash, malformed response) so a lexicon hiccup can never
        stall or crash an hours-long, unattended game run. Every call
        (success or failure) is logged if call_log_path was configured.

        The lens (server default: on) is left enabled -- see module docstring.
        Timeout defaults to 30s, comfortably past lexicon's own 20s lens
        backstop plus corpus-load and IPC overhead.
        """
        call_args = dict(args)

        start = time.monotonic()
        result: Optional[dict] = None
        error: Optional[str] = None
        try:
            if self._proc is None:
                raise RuntimeError("lexicon client not started")
            async with self._lock:
                raw = await self._request("tools/call", {"name": name, "arguments": call_args}, timeout=timeout)
            content = (raw or {}).get("content") or []
            text = content[0].get("text") if content and isinstance(content[0], dict) else None
            result = json.loads(text) if text else None
            if isinstance(raw, dict) and raw.get("isError"):
                error = f"tool reported isError: {text}"
                result = None
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.warning(f"[lexicon_client] {name} call failed: {error}")

        gate_dropped: list = []
        if result is not None and name in _GATED_TOOLS:
            result, gate_dropped = await self._materiality_filter(call_args.get("text", ""), result)

        latency_ms = (time.monotonic() - start) * 1000
        self._log_call(name, call_args, latency_ms, result, error, gate_dropped=gate_dropped, power=power)
        return result

    async def _materiality_filter(self, scenario: str, result: dict) -> tuple[dict, list]:
        """Filters a lens-capable tool's returned atoms down to the ones that
        genuinely bear on this scenario, not merely share its vocabulary or
        topic -- see LEXICON_MATERIALITY_PROMPT. Fails OPEN: any error here
        (bad JSON, API failure, timeout) returns the original, unfiltered
        result rather than risk silently hiding a genuinely good match behind
        a broken gate. Returns (possibly-filtered result, dropped atom
        summaries) for logging."""
        key, items = _atoms_field(result)
        if not items:
            return result, []

        candidates = "\n".join(
            f"- id={a['id']}: {a['name']} -- {a.get('agent_instruction') or a.get('mechanism') or a.get('gloss') or ''}" for a in items
        )
        prompt = LEXICON_MATERIALITY_PROMPT.format(scenario=scenario[:2000], candidates=candidates)
        try:
            response = await self._gate_client.messages.create(
                model=LEXICON_MATERIALITY_GATE_MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Haiku doesn't reliably return bare JSON despite "No other text" --
            # observed both a ```json fence and trailing prose after the array in
            # separate runs. Extract the array substring rather than assume the
            # whole response is clean JSON.
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                raise ValueError(f"no JSON array found in gate response: {text[:200]!r}")
            passing_ids = set(json.loads(match.group(0)))
        except Exception as e:
            logger.warning(f"[lexicon_client] materiality gate failed, passing all atoms through unfiltered: {type(e).__name__}: {e}")
            return result, []

        kept = [a for a in items if a.get("id") in passing_ids]
        dropped = [_summarize_atom(a) for a in items if a.get("id") not in passing_ids]
        filtered_result = dict(result)
        filtered_result[key] = kept
        return filtered_result, dropped

    def _log_call(
        self,
        name: str,
        args: dict,
        latency_ms: float,
        result: Optional[dict],
        error: Optional[str],
        gate_dropped: Optional[list] = None,
        power: Optional[str] = None,
    ) -> None:
        if not self.call_log_path:
            return
        entry: dict[str, Any] = {
            "ts": time.time(),
            "power": power,
            "tool": name,
            "args": args,
            "latency_ms": round(latency_ms, 1),
            "ok": error is None,
            "error": error,
        }
        if result:
            _, items = _atoms_field(result)
            if items is not None:
                entry["atoms"] = [_summarize_atom(item) for item in items]
        if gate_dropped:
            entry["gate_dropped"] = gate_dropped
        try:
            with open(self.call_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            logger.warning("[lexicon_client] failed to write call log", exc_info=True)
