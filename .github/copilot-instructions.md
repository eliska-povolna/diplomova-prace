# Copilot Instructions for Development

## Commit Style
- **Batch changes**: Group related changes into fewer, larger commits rather than committing frequently
- **Brief messages**: Use concise 1-line commit messages during early development (e.g., "Add precomputation to training workflow")
- **Avoid narration**: Don't describe every small fix - focus on what changed at a high level
- **No detailed breakdowns**: Skip bullet-point commit message details unless truly necessary
- **During early development**: Commit once per feature/major task, not per file - batch all related changes together

## Development Approach
- During early phases: Prioritize speed and iteration over granular documentation
- Combine multiple related file changes in single commits
- Only create detailed documentation when requested or at milestone completions
- Keep the workflow focused on implementation, not documentation

## File Management
- Don't create separate documentation files for each change
- Avoid creating intermediate reference notebooks (integrate into main workflows instead)
- Update existing markdown files (like IMPLEMENTATION_PLAN.md) rather than creating new ones

## General Principles
- Think in terms of features/features, not individual lines of code
- Group logically related changes together
- Balance visibility with developer focus

## File Organization

### Analysis Documents
All analysis, investigation, and diagnostic markdown files should be placed in `docs/analysis/` directory, not in the repository root. This keeps the root clean and organizes project documentation hierarchically.

**Files that belong in `docs/analysis/`:**
- Deep-dive analyses (e.g., loss function analysis, architectural decisions)
- Investigation reports (e.g., performance bottlenecks, experimental results)
- Diagnostic documents (e.g., error patterns, debugging findings)
- Detailed explanations of complex components

**Examples:**
- ✅ `docs/analysis/SAE_LOSS_ANALYSIS.md` - Why loss values are within expected ranges
- ✅ `docs/analysis/AUDIT_REPORT.md` - Code review and architectural audit
- ✅ `docs/analysis/IMPLEMENTATION_SUMMARY.md` - Detailed implementation notes

**Root-level files that stay:**
- `README.md` - Project overview and setup
- `QUICK_START.md` - Getting started guide
- `TRAINING_README.md` - Training documentation
- `LICENSE` - License information

## Data Handling - CRITICAL

### ⛔ NEVER use placeholder or mock data without explicit user approval

When encountering data loading issues:

**REQUIRED STEPS (in order):**
1. **STOP** - Do not proceed with placeholder data
2. **DIAGNOSE** - Investigate why real data won't load (missing files, wrong paths, format issues)
3. **INFORM** - Tell Eliška exactly what the problem is and what you found
4. **ASK** - Ask how to proceed (load from alternate source? fix file path? skip for now?)
5. **EXECUTE ONLY IF APPROVED** - Only use mock/placeholder data if Eliška explicitly says "yes, use test data"

**When data loading fails:**
- ❌ DO NOT: Generate random categories, create dummy business entries, use default/sample values
- ✅ DO: Print error details, show what you tried, ask user for guidance

**Why this matters:**
- Placeholder data corrupts analysis and thesis quality
- It's dishonest to the research (thesis is about REAL Yelp data)
- It makes results unreplicable and potentially misleading
