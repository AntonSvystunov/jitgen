# The JitGen Method

---

## 1. What the method is for

The default way to run code produced by a language model is **generate-then-execute**: the model emits a complete snippet, and only then is that snippet handed to an interpreter. Two costs follow directly from that ordering.

1. **Additive latency.** One step of an agentic loop costs generation time *plus* execution time, even though most of the program was finished long before the last token appeared.
2. **Late error detection.** Syntax, semantic, and runtime errors surface only after the whole snippet exists — including the tail that was never going to run.

**JitGen removes the ordering constraint.** It interleaves parsing and execution *with* generation: every top-level statement is sent to a live interpreter as soon as its boundary is provably fixed, and consumption of the token stream stops at the first error.

The two things the method must deliver:

- **Lower loop latency** — first visible output appears at the first executable statement instead of at the end of the stream; on the failure path the run stops after consuming only a prefix.
- **Early error detection** — parse, semantic, and runtime errors are surfaced as soon as the relevant prefix is complete.

And the two things it must *not* cost:

- **Correctness.** For error-free programs the observable behaviour is identical to running the whole program afterwards.
- **Generality.** The method is model- and language-agnostic. It does not touch decoding, does not mask or bias sampling, does not require fine-tuning, and needs no access to model internals. It treats the model as an opaque source of a token stream.

### Explicit non-goals

- JitGen is **not** constrained decoding. It never intervenes in token sampling.
- JitGen does **not** repair, rewrite, or validate the model's intent. It executes what was generated, earlier.
- JitGen does **not** define the sandbox policy. Isolation is a deployment concern, configured by the embedder.

---

## 2. Core idea in one paragraph

Keep an append-only buffer of everything the model has emitted but that has not yet been executed. On each new fragment, append it and try to parse the buffer against the grammar of the target language. If the parse is incomplete, wait for more input. If the parse succeeds and yields `k ≥ 2` top-level statements, the first `k − 1` of them can no longer be changed by anything the model emits later — dispatch them to the interpreter, in order, and keep only the last one in the buffer. When the stream ends, parse and execute whatever is left. Any parse or execution error halts the run.

The whole method rests on one structural fact: **a completed top-level statement shields everything to its left.** New lexemes can only extend the last statement or open new ones to its right.

---

## 3. Principles

These are the invariants. A change that breaks one of them is a change to the method, not an optimisation of it.

### P1 — Boundaries come from the grammar, never from surface heuristics

Statement boundaries are determined by a parser over the target language's context-free grammar. Splitting on newlines, regex matching, brace counting, or indentation heuristics are all wrong: newline splitting misclassifies multi-line constructs and hands incomplete syntax to the interpreter, and the token boundaries of a subword vocabulary do not coincide with the lexeme boundaries of the language. A single identifier is usually several tokens; a single token may span a closing parenthesis and a colon.

### P2 — Dispatch only what is stable

When a successful parse of the buffer yields `k ≥ 2` top-level statements, dispatch exactly the first `k − 1`. **Never dispatch the last statement during the streaming phase** — it is still growing. It leaves the buffer only in the flush phase, after the stream has ended.

### P3 — A boundary counts as fixed only when it is self-delimiting

A statement's boundary is fixed only after a lexeme past which no continuation can belong to that statement. This is stricter than "the parser accepted it." For languages with compound statements it is essential: an `else:` block written after an `if x:` block *attaches to the previous statement* and moves its boundary. In Python this role is played by the layout (indentation) lexemes. Until that closing lexeme arrives, the construct is the still-growing last statement, not a fixed one.

### P4 — Semantic equivalence with post-hoc execution

If a stream is consumed in full without error, the sequence of values yielded must equal the sequence produced by executing the top-level statements of the complete program in order, and the final interpreter state must equal that of post-hoc execution. Concretely: **each statement runs exactly once, in left-to-right order, in a single interpreter session.**

### P5 — One persistent interpreter per stream

State (bindings, imports, side effects) must carry across dispatched statements. Executing each statement in a fresh process breaks P4.

### P6 — The buffer invariant

At the start of every iteration of the main loop, the buffer contains *exactly* the suffix of the tokens emitted so far that has not yet been dispatched for execution. Every code path — dispatch, incomplete parse, flush — must restore this invariant. This is what guarantees no statement is executed twice and none is dropped.

### P7 — Text-level interface, monotone growth

The model's tokens and the language's lexemes are compared as **strings**, not as units. Detokenisation is concatenation, so appending a fragment only extends the accumulated string on the right; this is what makes incremental parsing sound. When a fragment ends in the middle of a lexeme, the parser reports **Incomplete** and the algorithm waits for more input. `Incomplete` is a normal control state, not an error, and must never be reported to the caller as a failure.

### P8 — Fail fast, then halt

The first parse error, execution error, or error during the post-stream flush stops consumption of the stream. The method does not skip a failing statement and continue. The concrete halting mechanism (cancelling generation, closing the stream) depends on the runtime, but the semantics are fixed: no statement after the failure point is executed.

### P9 — Flush the remainder

After the stream ends, the buffer is parsed once more and every top-level statement it contains is executed. Without this step the last statement — deliberately withheld by P2 — would never run.

### P10 — Extraction is a separate, deterministic pre-step

Raw model output may carry code fences, prose, reasoning, or tool-call envelopes. Turning that into a plain string of the target language is a **pre-step outside the method's formal guarantees**. Keep it isolated, keep it deterministic, and keep it out of the core loop. All guarantees in this document concern the token stream of the *extracted* code.

### P11 — Execution is isolated

The interpreter runs in a sandbox. The method executes generated code earlier than usual, so it inherits every risk of executing generated code at all — and offers no protection against it by itself.

---

## 4. Preconditions

Before the method is applied to a new model, language, or parser, these must be checked. They are the boundary of validity, not optional hardening.

| Precondition | What it requires | How it is checked |
|---|---|---|
| **Suitability** of the model/language pair | Every string that splits into legal lexemes of the language must also be expressible as some sequence of model tokens | Tokenizer round-trip: `detok(tok(s)) == s`, byte for byte, over samples covering every lexeme class of the language |
| **Admissible parser** | Deterministic (one parse tree, no parse forest), valid-prefix property (errors reported at the earliest impossible lexeme), streaming (extends the parse per lexeme, no need for the whole input) | Property of the chosen parser class — LR(k)/LALR(1)/SLR and LL(k) qualify; general context-free parsers (Earley/GLR) do not, absent a disambiguation policy |
| **Stmt-grammar** | The start symbol derives only a possibly empty concatenation of the `Stmt` non-terminal, with no other symbols between statements | Grammars with explicit separators reduce to this form by a boundary-preserving normalisation that absorbs separators into a derived non-terminal |
| **Self-delimiting boundaries** | Each top-level statement has a closing lexeme past which nothing can join it | Language-specific; for Python, the layout lexemes |

The suitability condition is a **coverage** condition, not a bijection, and it says nothing about whether the model will actually emit a given sequence under given decoding settings — that is a separate question. For byte-level fallback vocabularies its practical content is **token-level losslessness**: violations come from information loss during encoding (lossy Unicode normalisation, collapsed control characters, whitespace normalisation that destroys the meaning of indentation), not from missing symbols.

---

## 5. Consequences contributors must respect

**Early halt changes error-handling behaviour.** If a top-level statement raises, and a `try/except` that would have caught it appears *later* in the stream, JitGen propagates the error and stops — the handler did not exist yet. This is specified behaviour, a direct reading of the correctness guarantee in the failure direction, not a bug. Downstream consumers that rely on whole-program error handling need to know this.

**Gains are bounded by the shape of the stream.** First-output latency improves whenever the first executable statement appears before the end of the stream. Total-time gains additionally need either an early failure or an executable tail long enough to overlap with generation. A stream that is mostly reasoning prose with code at the end has little to overlap. Do not treat a small total-time delta as a defect of the algorithm.

**Re-parsing has a cost.** Stateless re-parsing of the whole buffer on every fragment is quadratic in the worst case (short fragments, long statements). A truly incremental parser with cost amortised per fragment would make it linear. Execution cost is unaffected — each statement runs once regardless of fragment granularity — and buffer memory is bounded by the longest top-level statement, independent of how the stream is chunked.

---

## 6. Glossary

Use these terms exactly. Do not introduce synonyms in code, comments, issues, or docs.

| Term | Meaning |
|---|---|
| **Token stream** | The finite or infinite sequence of tokens emitted by the model, over its token vocabulary. The model is treated as an opaque source of this stream. |
| **Fragment** | One increment appended to the buffer from the stream. Fragment granularity is a property of the transport, not of the method. |
| **Detokenisation** | Concatenation of token surface forms into a string. Monotone over stream prefixes — this is what makes incremental parsing sound. |
| **Lexeme** | The parser's input unit: a lexeme class paired with a surface form. Corresponds to a terminal of the grammar. |
| **Surface form** | The character-level string of a token or lexeme. Tokens and lexemes are compared through their surface forms, never directly. |
| **Suitability** | The criterion on a model/language pair: every string that splits into legal lexemes of the language also splits into some sequence of model tokens. Verified as a tokenizer round-trip. |
| **Top-level statement** | A statement whose parse-tree node is a *direct child of the root*. These are the units of dispatch. Statements nested inside compound constructs are not top-level and run only as part of their enclosing statement. |
| **Admissible parser** | A parser that is deterministic, has the valid-prefix property, and is streaming. See §4. |
| **Valid-prefix property** | Reading lexemes left to right, the parser reports an error at the first lexeme after which the consumed prefix has no possible continuation in the language. Source of the earliest-possible error detection. |
| **Stmt-grammar** | A grammar whose start symbol derives only a possibly empty concatenation of `Stmt`, with no terminals or other non-terminals between statements. |
| **Boundary stability** | The property that, once a program prefix parses into `k` top-level statements, no continuation of the stream can change the first `k − 1` of them. The justification for the dispatch rule. |
| **Self-delimiting boundary** | A statement boundary that is fixed by a closing lexeme, past which no later lexeme can join that statement. Required for boundary stability to hold rather than be assumed. |
| **Incomplete** | The parser state meaning "valid so far, needs more input" — typically a fragment ending mid-lexeme, or an unclosed construct. A control state, never an error. |
| **Buffer** (unexecuted-suffix buffer) | The accumulated string of emitted-but-not-yet-dispatched tokens. Governed by the buffer invariant (P6). |
| **Dispatch** | Sending one fixed top-level statement to the interpreter for execution. |
| **REPL / interpreter** | The single persistent read-eval-print-loop session that receives dispatched statements and holds state across them. |
| **Streaming phase** | The main loop: append fragment → parse → dispatch the stable prefix. |
| **Flush** | The post-stream phase that parses and executes the remainder of the buffer. |
| **Asynchronous strategy** | Execution interleaved with generation — i.e. JitGen. |
| **Synchronous strategy** (post-hoc execution) | The baseline: consume the stream in full, then execute the complete program. The reference semantics for the correctness guarantee. |
| **Time to first output** | Wall-clock time from the start of the stream to the first value produced by the interpreter. |
| **Code extraction pre-step** | The deterministic stage that reduces raw model output to a plain string of the target language. Outside the formal guarantees (P10). |
| **Agentic loop** | The control loop in which a model emits code, an environment executes it, and the result re-enters the model's context as an observation. The setting the method targets. |

---

## 7. Review checklist for changes

Before merging a change to the core loop, confirm:

- [ ] The buffer invariant (P6) holds on every path — dispatch, incomplete, error, flush.
- [ ] No statement can be dispatched twice, and none can be dropped between the streaming phase and the flush.
- [ ] The last recognised statement is never dispatched during streaming (P2).
- [ ] Boundary detection still goes through the parser, with no surface-level shortcut added for speed (P1).
- [ ] `Incomplete` is handled as a control state and never escapes as an error (P7).
- [ ] The first error halts consumption; nothing after it executes (P8).
- [ ] Statement order is preserved and the interpreter session is shared (P4, P5).
