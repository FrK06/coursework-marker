# Page Citation Fix

## The Problem

1. **DOCX files showed wrong page count**: 11 actual pages → 7 estimated
2. **LLM cited incorrect page numbers**: Because estimates were wrong, citations were wrong

## Root Cause

DOCX files don't have real page numbers (they're flow documents). The old code used character count (~2500 chars/page) to estimate, which was inaccurate.

## The Fix

**For DOCX files**: Don't show page numbers at all - use ONLY section numbers for citations
**For PDF files**: Show accurate page numbers (PDF has real pages)

## Files to Replace

| File | What Changed |
|------|--------------|
| `src/document_processing/docx_processor.py` | Added `pages_are_accurate=False` flag, reduced CHARS_PER_PAGE to 1800 |
| `src/document_processing/pdf_processor.py` | Added `pages_are_accurate=True` flag |
| `src/retrieval/retriever.py` | `format_context_for_llm()` now hides page numbers for DOCX |
| `src/prompts/ksb_templates.py` | Updated citation rules: section-based for DOCX, page+section for PDF |
| `ui/ksb_app.py` | Tracks `pages_are_accurate`, shows warning in UI for DOCX estimates |

## How It Works Now

### For DOCX (pages_are_accurate=False):
- UI shows: `Pages (est.): ~7` with warning
- Evidence headers: `(Section 3)` - no page numbers
- Prompt tells LLM: "Do NOT cite page numbers - they are not available"

### For PDF (pages_are_accurate=True):
- UI shows: `Pages: 11` - accurate
- Evidence headers: `(Section 3 / page 7)` - both shown
- Prompt tells LLM: "Cite using Section AND page numbers"

## Installation

Replace these 5 files in your project:
```
src/document_processing/docx_processor.py
src/document_processing/pdf_processor.py
src/retrieval/retriever.py
src/prompts/ksb_templates.py
ui/ksb_app.py
```

## Expected Results

After this fix:
- LLM will cite by section only for DOCX files (e.g., "Section 3", "Section 5")
- LLM will cite by section AND page for PDF files (e.g., "Section 3, page 7")
- No more incorrect page citations for DOCX documents
