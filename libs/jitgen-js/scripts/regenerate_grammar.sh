#!/usr/bin/env bash
# Regenerates jitgen_js/_grammar/*.py from the vendored .g4 sources.
#
# Requires a Java runtime and the ANTLR 4.13.2 complete jar locally. NOT
# required by consumers of the published jitgen-js package — only by
# contributors regenerating the grammar itself.
#
# Usage: scripts/regenerate_grammar.sh [path-to-antlr-4.13.2-complete.jar]
set -euo pipefail

ANTLR_JAR="${1:-$HOME/.antlr/antlr-4.13.2-complete.jar}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAMMAR_DIR="$(cd "$SCRIPT_DIR/../jitgen_js/_grammar" && pwd)"

test -f "$ANTLR_JAR" || {
    echo "ANTLR jar not found at $ANTLR_JAR" >&2
    echo "Download it from https://www.antlr.org/download/antlr-4.13.2-complete.jar" >&2
    exit 1
}

# 1. Apply the Java/C#-style `this.` -> Python `self.` predicate rewrite
#    (transformGrammar.py comes from antlr/grammars-v4's
#    javascript/javascript/Python3/ subdirectory; a pinned copy lives at
#    scripts/vendor/transformGrammar.py for reproducibility). Only needed
#    when re-fetching fresh .g4 files from grammars-v4 — the vendored .g4
#    files under jitgen_js/_grammar/ already have this rewrite applied, so
#    running it again against them is a harmless no-op (nothing left
#    matching `this.` to replace), except that it also overwrites the
#    `.g4.bak` backups; skip this step entirely if you haven't replaced the
#    .g4 files with fresh copies from upstream.
(cd "$GRAMMAR_DIR" && python3 "$SCRIPT_DIR/vendor/transformGrammar.py")

# 2. Generate the lexer/parser/listener/visitor directly into _grammar/.
java -jar "$ANTLR_JAR" -Dlanguage=Python3 -visitor -o "$GRAMMAR_DIR" \
    "$GRAMMAR_DIR/JavaScriptLexer.g4" "$GRAMMAR_DIR/JavaScriptParser.g4"

# ANTLR 4.13.2's Python3 target already emits `if "." in __name__: from
# .JavaScriptParserBase import ... else: from JavaScriptParserBase import
# ...` in every generated file needing a sibling import, so no post-generation
# import-rewrite step is needed for this ANTLR version — confirmed by
# inspecting the generated output, not assumed. Re-verify this if the pinned
# ANTLR_TOOL_VERSION below is ever bumped.

echo "Regenerated. Review the diff, then run: uv run pytest"
