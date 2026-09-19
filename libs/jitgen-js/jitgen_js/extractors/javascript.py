from antlr4 import CommonTokenStream, InputStream, Token
from antlr4.error.Errors import ParseCancellationException, RecognitionException
from antlr4.error.ErrorStrategy import BailErrorStrategy
from antlr4.ParserRuleContext import ParserRuleContext
from jitgen.base import SourceCode

from jitgen_js._grammar.JavaScriptLexer import JavaScriptLexer
from jitgen_js._grammar.JavaScriptParser import JavaScriptParser

# Statement kinds whose grammar rule has an optional *trailing* clause that
# only a later chunk can prove absent — an `if` without an `else` yet, a
# `try/catch` that might still grow a `finally`. Releasing one of these the
# moment it parses would risk handing the executor a fragment the model was
# not actually finished with; the N-1 rule defers each until a following
# sibling statement starts, which naturally means the optional clause turned
# out not to belong to it after all. Loops and labelled statements have no
# such optional clause in this grammar, but wrap an arbitrary nested
# `statement` whose own sealedness this v1 does not recurse into; treating
# them as extensible too is a conservative (never incorrect, only a little
# slower) simplification instead of implementing that recursion.
_EXTENSIBLE_CONTEXT_NAMES: frozenset[str] = frozenset(
    {
        "IfStatementContext",
        "TryStatementContext",
        "WhileStatementContext",
        "ForStatementContext",
        "ForInStatementContext",
        "ForOfStatementContext",
        "LabelledStatementContext",
    }
)

# Characters whose presence right after a statement proves a *real* terminator
# was already in the model's own output — as opposed to the synthetic EOF
# ANTLR's `eos` rule also accepts as a statement terminator (see
# `_is_sealed`'s docstring for why that distinction matters).
_SEAL_MARKERS: tuple[str, ...] = (";", "\n", "}")


class JavaScriptAntlrExtractor:
    """ANTLR4-grammar statement extractor for JavaScript.

    Unlike `LarkStatementExtractor`/`PythonLarkExtractor`, this is a single,
    self-contained class rather than a base/subclass split: it is the only
    ANTLR-based extractor in this monorepo, for the one grammar this package
    exists to support, so there is no second caller to justify factoring out
    a shared base yet. It reimplements the same N-1/sealing *design* as the
    Lark extractors, adapted to ANTLR4's `ParserRuleContext` API in place of
    Lark's `Tree`/`meta`.

    The grammar's own `eos` (end-of-statement) rule accepts a synthetic EOF
    as a valid terminator (`eos : SemiColon | EOF | {lineTerminatorAhead}? |
    {closeBrace}? ;`), so parsing a truncated buffer like `"x = 1"` alone
    *succeeds* as one complete statement even though the model may still be
    mid-expression. This is harmless for every statement except the very
    last one in the buffer — a statement with a sibling after it could only
    have closed via a real terminator, never the synthetic EOF — which is
    exactly the one statement the N-1 rule already treats specially via
    `_is_sealed`. `_is_sealed` is therefore written to check for a genuine
    trailing terminator in the raw source text, never "the parse tree says
    this statement is complete".

    A grammar rule wraps every top-level statement two levels deep
    (`program -> sourceElements -> sourceElement -> statement -> <concrete
    statement>`); `sourceElements`/`sourceElement` are transparent
    single-child wrappers with no separators between them, so `Stmt-grammar`
    (METHOD.md §4) holds once dispatch units are read from
    `program.sourceElements().sourceElement()` rather than from `program`'s
    direct children.
    """

    def _build_parser(self, source: SourceCode) -> JavaScriptParser:
        """Build a fresh lexer/token-stream/parser for one parse attempt.

        ANTLR's generated `Lexer`/`Parser` are cheap to construct (all the
        expensive work — building parsing tables — already happened once,
        ahead of time, when the grammar was generated) but are stateful
        across a single input; the idiomatic pattern is a fresh instance per
        parse, unlike Lark's stateless, cacheable `Lark.parse`.

        `removeErrorListeners()` on both lexer and parser suppresses
        ANTLR's default `ConsoleErrorListener`, which otherwise writes every
        syntax error straight to stderr — and a mid-statement parse *looks*
        like a syntax error on nearly every chunk while streaming, since
        `extract` re-parses the whole buffer on each call.

        `BailErrorStrategy` replaces ANTLR's default silent
        skip/insert recovery, which would otherwise hand back a tree ANTLR
        patched over rather than a faithful parse of what the model wrote.

        Args:
            source: The full unexecuted-suffix buffer accumulated so far.

        Returns:
            A `JavaScriptParser` ready to have `.program()` called on it.
        """
        lexer = JavaScriptLexer(InputStream(source))
        lexer.removeErrorListeners()
        stream = CommonTokenStream(lexer)
        parser = JavaScriptParser(stream)
        parser.removeErrorListeners()
        parser._errHandler = BailErrorStrategy()
        return parser

    def _parse_or_recover(
        self, source: SourceCode, *, final: bool
    ) -> JavaScriptParser.ProgramContext | None:
        """Attempt to parse `source`; every failure is recoverable until `final`.

        This is the disambiguation policy METHOD.md §4 requires in place of
        the "valid-prefix property" its Admissible-parser precondition
        assumes: ANTLR4's ALL(*) prediction does not have one. `PythonLarkExtractor`
        can tell "incomplete" from "wrong" by checking whether the
        offending token sits at the buffer's tail, because Lark's LALR
        lexer/parser fails exactly where the truncation is. ALL(*) does not:
        it explores the token stream with unbounded lookahead *before*
        committing to an alternative, so when that exploration runs out of
        real tokens, the exception it raises blames whatever token the
        *outer* rule invocation started from — which can be arbitrarily far
        from the buffer's tail. Two concrete failures found by feeding this
        extractor real streamed JS chunk-by-chunk (neither caught by a
        tail-position check, both fixed by removing the check entirely):

        - `'try {\\n  JSON.parse("{'` (an unterminated string containing a
          `{`): the grammar's lexer routes the unmatched opening quote to a
          hidden `ERROR` channel (see `JavaScriptLexer.g4`'s catch-all
          `UnexpectedCharacter` rule) and then lexes the `{` *fresh*, as a
          real `OpenBrace` token, which derails argument-list prediction —
          ANTLR reports `NoViableAltException` pointing at the `(` of
          `JSON.parse(`, several tokens before the truncation.
        - `'...console.log("caught: " + e.n'` (an ordinary unclosed call,
          mid-argument): `InputMismatchException` pointing at `console.log`'s
          own `(`, not at the truncated `e.n`.

        Both are just the unclosed-parenthesis/brace case that streaming
        code hits constantly, not rare corner cases — so the only sound
        policy is to treat *every* parse failure as "more input needed"
        while streaming, and rely exclusively on `final=True` to catch a
        genuinely invalid buffer. The corresponding trade-off, made
        explicit here since CLAUDE.md advertises early error surfacing as
        the method's headline benefit: a JS syntax error is only ever
        raised at end-of-stream flush, never mid-stream — `Session.push`'s
        `has_error` (which callers poll to abort the model stream early)
        will not trip for a JS syntax error the way it does for Python's.

        Args:
            source: The full unexecuted-suffix buffer accumulated so far.
            final: `True` when the stream has ended, so a parse error can no
                longer be explained away as "more input is still coming".

        Returns:
            A `ProgramContext` on successful parse; `None` when `not final`
            and the parse failed for any reason.

        Raises:
            SyntaxError: when `final` and the parse still fails.
        """
        parser = self._build_parser(source)
        try:
            return parser.program()
        except ParseCancellationException as exc:
            if not final:
                return None
            cause = exc.args[0] if exc.args else None
            token = getattr(cause, "offendingToken", None) if cause else None
            raise SyntaxError(self._describe(cause, token)) from exc

    @staticmethod
    def _describe(cause: RecognitionException | None, token: Token | None) -> str:
        """Build a human-readable message from a `RecognitionException`.

        `RecognitionException` carries no useful `__str__`/`.message` in the
        Python runtime — `str(cause)` renders as the unhelpful literal
        `"None"` — so the message is built from the offending token instead.

        Args:
            cause: The `RecognitionException` ANTLR raised, or `None`.
            token: `cause`'s offending token, or `None`.

        Returns:
            A message naming the offending token and its position, or a
            generic fallback when neither is available.
        """
        kind = type(cause).__name__ if cause is not None else "syntax error"
        if token is None:
            return kind
        if token.type == Token.EOF:
            return f"{kind}: unexpected end of input"
        return f"{kind}: unexpected {token.text!r} at line {token.line}:{token.column}"

    def _is_sealed(self, source: SourceCode, ctx: ParserRuleContext) -> bool:
        """Check whether `ctx` could still be extended by input not yet arrived.

        A statement is sealed (safe to release immediately, without waiting
        for a following sibling statement to start) iff both hold:

        * its concrete grammar rule cannot take a further trailing clause —
          `_EXTENSIBLE_CONTEXT_NAMES` names the ones that can; and
        * a real `;` or `}` terminates it, or a real newline follows it, in
          `source`.

        The second check is what makes this safe against the `eos`-EOF trap
        documented on the class. Empirically (confirmed by parsing, not
        assumed): when `eos` matches via its synthetic-`EOF` alternative,
        `ctx.stop` is *not* the EOF token — it remains whatever real content
        token came before it (e.g. the `1` in `"x = 1"`), so checking
        `ctx.stop.type` cannot distinguish this case. What *does* distinguish
        it is that nothing beyond that real content token exists in `source`
        at all: `source[ctx.stop.stop:]` is just that token's own last
        character with nothing after it, containing none of `;`/`\\n`/`}`.
        Compare a genuinely terminated statement: for `"do {x()} while(c);"`,
        `ctx.stop` *is* the `;` token itself, so `source[ctx.stop.stop:]`
        starts with `;` and this correctly returns `True` even though
        nothing follows the semicolon either — the terminator's own presence
        is what counts, not whatever (if anything) comes after it. Scanning
        from `ctx.stop.stop` (inclusive of the terminator's own character)
        rather than `ctx.stop.stop + 1` is what makes both cases fall out of
        one check.

        Args:
            source: The full unexecuted-suffix buffer the parse tree came from.
            ctx: The concrete statement context to check — one level below
                `sourceElement`/`statement`'s wrapper contexts (see
                `_concrete_statement`).

        Returns:
            `True` if `ctx` is sealed and safe to release immediately.
        """
        if type(ctx).__name__ in _EXTENSIBLE_CONTEXT_NAMES:
            return False
        # Inclusive of ctx.stop's own last character — see the docstring
        # above for why that character alone (a real `;`/`}`) is already
        # sufficient proof, with nothing required to follow it.
        scan_from = ctx.stop.stop if ctx.stop is not None else len(source)
        tail = source[scan_from:]
        return any(marker in tail for marker in _SEAL_MARKERS)

    @staticmethod
    def _concrete_statement(source_element: ParserRuleContext) -> ParserRuleContext:
        """Unwrap a `sourceElement` context down to its concrete statement node.

        `sourceElement : statement ;` and `statement` is itself an unlabeled
        wrapper alternative, so the node whose class name identifies the
        actual statement kind (`IfStatementContext`, `ExpressionStatementContext`,
        ...) sits two levels below `sourceElement` — checking
        `type(source_element).__name__` directly would always see
        `SourceElementContext` and silently defeat `_is_sealed` for every
        statement kind.

        Args:
            source_element: One `SourceElementContext` from `program.
                sourceElements().sourceElement()`.

        Returns:
            The concrete statement context `source_element` wraps.
        """
        return source_element.statement().getChild(0)

    @staticmethod
    def _is_ambiguous_tail_fragment(
        source: SourceCode, element: ParserRuleContext
    ) -> bool:
        """Check whether `element` is a single token that might still be growing.

        A truncated multi-character keyword lexes as a complete, ordinary
        token for whatever characters are available — ANTLR's maximal-munch
        lexer has no way to know more characters might follow in a later
        chunk — and when that fragment happens to form a valid token on its
        own, it does not merely cause a parse *error* to defer: it parses as
        an entirely different, genuine top-level statement. Confirmed
        concretely: buffer `"...try {\\n
        JSON.parse(...); \\n} catch (e) {...}\\n final"` — the not-yet-complete
        `finally` keyword lexes as identifier `final`, which parses as its
        own one-token `expressionStatement`. `_is_sealed` correctly refuses
        to seal that trailing `final` element (nothing follows it in
        `source`), but N-1's *other* guarantee — "a further child could only
        ever extend the last one" — is what actually breaks: the preceding
        `TryStatementContext` is still extensible, and if the model's next
        chunk completes `finally { ... }`, a fresh parse would attach it to
        that *same* try-statement rather than leave `final`/`finally` as an
        independent element at all. Releasing the try-statement here would
        hand the executor a `try/catch` with no `finally`, permanently
        losing the block the model was about to attach.

        `extract` treats an element this flags as not existing yet — as if
        the buffer had stopped one token earlier — so N-1 and `_is_sealed`
        are applied to whatever precedes it instead.

        Args:
            source: The full unexecuted-suffix buffer the parse tree came from.
            element: A `SourceElementContext` to check.

        Returns:
            `True` if `element` spans exactly one token and that token's end
            reaches `source`'s last character.
        """
        return element.start is element.stop and element.stop.stop >= len(source) - 1

    @staticmethod
    def _statement_text(source: SourceCode, ctx: ParserRuleContext) -> SourceCode:
        """Slice the source text spanned by `ctx` out of `source`.

        Args:
            source: The full unexecuted-suffix buffer the parse tree came from.
            ctx: The parse tree node to extract text for.

        Returns:
            The text `ctx` spans in `source`.
        """
        return source[ctx.start.start : ctx.stop.stop + 1]

    def extract(
        self, source: SourceCode, *, final: bool
    ) -> tuple[list[SourceCode], SourceCode]:
        """Extract ready top-level statements from `source`.

        Args:
            source: The full unexecuted-suffix buffer accumulated so far.
            final: `True` when the stream has ended, so every remaining
                statement is returned rather than the last one withheld as
                still-growing.

        Returns:
            A `(statements, leftover)` pair: statements ready to dispatch, in
            order, and the suffix of `source` not yet covered by a complete
            statement.

        Raises:
            SyntaxError: if `_parse_or_recover` raises.
        """
        if not source.strip():
            return [], ("" if final else source)

        program = self._parse_or_recover(source, final=final)
        if program is None:
            return [], source

        elements_ctx = program.sourceElements()
        elements = elements_ctx.sourceElement() if elements_ctx is not None else []

        if final:
            statements = [self._statement_text(source, el) for el in elements]
            return [s for s in statements if s.strip()], ""

        # A trailing element that's just one token reaching the buffer's tail
        # might still be a keyword mid-truncation rather than a genuine new
        # statement (see `_is_ambiguous_tail_fragment`) — treat it as not
        # existing yet so N-1/`_is_sealed` are evaluated against whatever
        # precedes it instead, rather than trusting it marks a real boundary.
        while elements and self._is_ambiguous_tail_fragment(source, elements[-1]):
            elements = elements[:-1]

        # Non-final: the first N-1 elements are complete by construction — a
        # further element could only ever extend the *last* one. The last
        # element is additionally safe when it cannot grow: see `_is_sealed`.
        if elements and self._is_sealed(source, self._concrete_statement(elements[-1])):
            ready = elements
        elif len(elements) < 2:
            return [], source
        else:
            ready = elements[:-1]

        statements: list[SourceCode] = []
        executed_upto = 0
        for element in ready:
            stmt = self._statement_text(source, element)
            if stmt.strip():
                statements.append(stmt)
            executed_upto = element.stop.stop + 1
        return statements, source[executed_upto:]
