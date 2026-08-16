# Workspace Project Instructions

These instructions apply to every project under this workspace. A `CLAUDE.md` closer to a project may add more specific architecture, commands, interfaces, and submission requirements.

## Scope and Priority

- Treat the user's current request as the active objective.
- Within a project, follow requirements in this order: assignment or product specification, assessment rubric or acceptance criteria, required public interfaces and file formats, project-level `CLAUDE.md`, then existing implementation details.
- When two sources conflict, do not silently choose one. Identify the conflict, explain its practical effect, and follow the higher-priority source.
- Required filenames, function signatures, schemas, protocols, and output formats are strict unless the governing requirement changes them.
- Do not apply conventions from one workspace project to another without confirming they match.

## Project Discovery

Before changing a project:

1. Identify the exact project root and check for a nearer `CLAUDE.md`, `AGENTS.md`, README, assignment specification, rubric, and contribution guide.
2. Inspect repository status and preserve existing uncommitted work.
3. Find the real entry points, dependency files, test layout, build scripts, and generated-file boundaries.
4. Read the files directly involved in the task and trace relevant producers and consumers.
5. Determine which checks can be run locally and which require network, credentials, special data, hardware, or manual assessment.

- Prefer evidence from repository files and runnable commands over assumptions based on typical project layouts.
- If the project root, publication target, or submission scope remains ambiguous and the choice would materially change the result, ask before acting.

## Working Method

- Make the smallest coherent change that fully satisfies the request.
- Extend existing implementation patterns instead of creating parallel systems.
- Preserve architecture, naming, formatting, package management, and dependency choices unless the task requires changing them.
- Keep unrelated cleanup and refactoring out of the task diff.
- Reuse existing scripts, fixtures, helpers, templates, and configuration.
- Do not overwrite, revert, reformat, or relocate unrelated user work.
- For large tasks, work in bounded stages: inspect, plan, implement, verify, then hand off.
- For diagnosis-only requests, provide the cause and evidence without implementing a fix unless requested.
- When blocked, exhaust safe read-only checks and in-scope alternatives before requesting user input.

## Commands and Environment

- Use the dependency manager and runtime already selected by the project.
- Derive install, run, test, lint, type-check, build, and packaging commands from project configuration or verified documentation.
- Do not invent a command merely because its name is conventional.
- Run commands from the directory expected by the project and use explicit paths when the location could be ambiguous.
- On Windows, prefer PowerShell-native commands and literal paths for filesystem operations.
- Keep temporary scripts, scratch analysis, and disposable artifacts in `work/`, a project-specific temporary directory, or the system temporary directory.
- Keep user-facing deliverables in the project's designated output or submission directory.
- Do not print secret environment-variable values; report presence or absence instead.

## Implementation Rules

- Preserve required public APIs and data contracts.
- When changing a shared interface, inspect all callers, consumers, tests, stored data, documentation, and migration implications.
- Do not add a dependency, framework, service, database, or architectural layer without a concrete benefit to the requested outcome.
- Prefer readable, direct implementations over speculative abstractions.
- Handle errors according to existing project conventions; do not hide failures behind unjustified fallback values.
- Comments should explain non-obvious decisions, constraints, formulas, or hardware behavior rather than narrating the code.
- Edit source files or generators instead of generated outputs when generation is available.
- Update nearby documentation when commands, configuration, interfaces, or externally visible behavior change.

## Testing and Verification

- Verify work in proportion to the risk and scope of the change.
- Start with the narrowest relevant test or check, then run broader checks for shared or cross-module changes.
- Add or update tests when behavior changes; add a regression test for a fixed defect when practical.
- Use the project's actual test, lint, type-check, build, formatting, simulation, or validation commands.
- Record which commands were actually run and whether they passed.
- If a check cannot run, state the missing dependency, environment, service, data, device, or permission.
- Never report a test, build, measurement, or manual inspection as passed unless it was executed or directly observed.
- A passing assertion proves only the behavior it checks; it does not establish complete correctness, material quality, calibration, robustness, or real-world performance.

## Coursework Requirements

For university and assessed projects:

- Treat the assignment specification and rubric as acceptance criteria.
- Maintain a clear mapping from each assessed requirement to its implementation, test, report section, and evidence.
- Respect prescribed filenames, standalone-file requirements, interfaces, templates, word limits, and submission formats exactly.
- Do not replace a minimal required submission with a larger project bundle.
- Keep student-authored decisions and reasoning visible; do not obscure important logic behind unnecessary generated abstractions.
- Identify assumptions, unresolved requirements, and items requiring lecturer or tutor confirmation.
- Before submission, inspect the final artifact independently of the development tree.

## Research and Experimental Integrity

- Never fabricate citations, datasets, logs, screenshots, measurements, benchmarks, statistical results, hardware observations, or test output.
- Keep raw data immutable and separate processed data and generated outputs.
- Record input data, preprocessing, configuration, parameters, random seeds, dependency versions, commands, environment, and code revision needed to reproduce an experiment.
- Use comparable datasets, environments, metrics, and conditions when comparing methods.
- Do not omit negative, null, flaky, or unexpected results merely because they weaken the preferred conclusion.
- Clearly label the evidence level:
  - implemented but not executed;
  - automated software test;
  - simulation or synthetic data;
  - live integration observation;
  - calibrated physical measurement;
  - inference or unverified target.
- Do not present connectivity or integration success as proof of accuracy, calibration, throughput, reliability, or physical performance.
- Keep unsupported physical or performance fields null or explicitly unverified until a valid measurement process exists.
- Prefer primary sources for technical and academic claims and confirm that every citation genuinely supports the associated statement.

## Reports and Documentation

- Keep README material for users and maintainers; keep agent-specific behavior in `CLAUDE.md`.
- Follow the required report language, template, citation style, page or word limit, and marking structure.
- Define symbols, units, assumptions, boundary conditions, and evaluation metrics.
- Give figures and tables meaningful captions and identify their data source, units, configuration, and experimental conditions.
- Keep conclusions within the strength of the available evidence.
- Distinguish proposed design, implemented system, verified behavior, and future work.
- Do not overwrite manually curated references, report sections, diagrams, or experimental records without inspecting them first.

## Data, Privacy, and Secrets

- Follow project data licences, ethics approval, consent, privacy, retention, and redistribution requirements.
- Do not commit credentials, API keys, tokens, private URLs, personal information, restricted datasets, or device secrets.
- Use environment files or an approved secret manager, and ensure private local files are ignored by Git.
- Use anonymized or synthetic examples when real data is unnecessary.
- Do not upload project files, data, or code to external services unless the user authorized that destination and scope.

## Git, Publication, and Submission

- Inspect repository status and relevant diffs before repository-wide changes and before handoff.
- Preserve existing remotes, especially teacher, upstream, and team remotes.
- Do not commit, push, publish, deploy, tag, open a pull request, or change repository visibility unless requested.
- Before publishing, confirm the exact project root, repository, branch, remote, visibility, and included file scope.
- Before submitting, verify required files, naming, format, generated artifacts, and exclusions.
- Never discard uncommitted work or rewrite shared history without explicit authorization.
- Keep commits focused and exclude temporary files, secrets, local environments, caches, and unrelated artifacts.

## Completion Standard

A task is complete only when:

1. The requested behavior or artifact is present.
2. Relevant requirements, interfaces, and file formats remain satisfied.
3. Appropriate verification was run, or missing validation is explicitly documented.
4. No unrelated user work was overwritten or reformatted.
5. Documentation and claims agree with the implementation and evidence.
6. Remaining hardware, calibration, external-service, manual, or assessor checks are listed.
7. The final handoff identifies the changed files, verification results, and important limitations.

## Project and Local Overrides

- Put project-specific architecture, exact commands, directory maps, style rules, and submission details in that project's `CLAUDE.md`.
- Put machine-specific paths, local device addresses, sandbox URLs, preferred test data, and other private preferences in `CLAUDE.local.md`.
- Add `CLAUDE.local.md` to `.gitignore` and never store credentials in it.
- For a repository already using `AGENTS.md`, a project `CLAUDE.md` may import it with `@AGENTS.md` and then add only Claude-specific instructions.
- Move detailed file-type or directory-specific guidance to `.claude/rules/<topic>.md`.

## Maintenance

- Add a rule when the same workspace-specific correction is needed more than once.
- Remove rules that are obsolete, duplicated, vague, or inconsistent with current projects.
- Review this file after major changes to workspace layout, course requirements, development tools, or publication practices.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes_tool` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review — token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.
