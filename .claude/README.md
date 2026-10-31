# Claude Code Multi-Agent Pack (native Markdown only)

Eight standalone subagents + one optional global orchestration instruction. No executable scripts, dependencies, Kilo keys or endpoints are bundled.

## Install into Termux

1. Unzip the archive into `~/claude-multiagent-pack`.
2. Create `~/.claude/agents` and copy the 8 `.md` files there (preserve existing files).
3. Copy `multi-agent-workflow.md` to `~/.claude/`.
4. Add one line `@multi-agent-workflow.md` to the bottom of `~/.claude/CLAUDE.md` (create it if absent), but only once.
5. Start a new Claude Code session (`ccr start --no-open` and `claude`) from the desired project directory.

## Why globally?
`~/.claude/agents` makes all 8 agents available in all projects. If you want project-only scope, copy them to `<project>/.claude/agents` and put the orchestration rules in that project's `CLAUDE.md` instead.

## Try

`Use the repo-explorer agent to map the current repository. No edits.`

`Use the researcher agent to check the official docs for the current version of my framework. Provide sources.`

`Have reviewer examine the proposed plan before coding.`

Type `@` in Claude Code and select an agent to force delegation for one task. `/agents` no longer opens the old wizard in current Claude Code versions.

## Notes
- All agents use `model: inherit` to avoid accidental CCR/Kilo model misrouting.
- read-only agents have no Bash/Edit/Write tools, and reviewer/architect/explorer/security use plan mode. An existing parent's permission mode can override agent mode, so keep permissions conservative.
- Coder and testers can cause filesystem side-effects; keep confirmation prompts enabled.
- Research MCP tool calls only work if the MCP is actually connected and available. ToolSearch, WebSearch/WebFetch availability through CCR is not guaranteed.
- `CLAUDE.md` is an instruction, not an enforced permission rule.
- Review every installed Markdown before asking Claude to execute it, as you should for any third-party agent definition.
