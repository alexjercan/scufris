# Add PDF preview extraction and source citations

- STATUS: OPEN
- PRIORITY: 0
- TAGS: feature,backlog,artifacts,pdf,frontend
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT

## Story

As a research or document user, I want PDFs previewed and their extracted text
linked back to page-level sources, so that an agent can cite evidence while I
can inspect the original document in the same workflow.

## Steps

- [ ] Select maintained PDF rendering and extraction libraries with Nix
      packaging, worker/sandbox, password, malformed-file, and license support.
- [ ] Add page rendering, thumbnails, page navigation, zoom, search, text
      selection, metadata, and download/export controls.
- [ ] Extract normalized page text and stable page/region citation anchors while
      retaining checksum and source provenance.
- [ ] Expose extracted content to authorized research tools without granting
      arbitrary filesystem access.
- [ ] Bound CPU, memory, page count, file size, nested object, and render time;
      provide explicit unsupported/encrypted/corrupt states.
- [ ] Add scanned, text, malformed, password-protected, oversized, RTL, and
      citation-roundtrip fixtures.

## Definition of Done

- Text and scanned PDF fixtures preview at desktop and mobile widths
  (test: `pdf-artifact-viewer.spec.ts`).
- An extracted citation navigates to the correct source page/region
  (test: `test_pdf_citation_roundtrip`).
- Malformed, encrypted, and resource-heavy fixtures fail within configured
  bounds without destabilizing Scufris (test: `test_pdf_processing_limits`).
- Citations remain understandable when copied into a research report (manual: user check).

## Notes

- Epic: 20260729-102210.
- Depends on: 20260729-102212.
- Record the PDF library and isolation choice in `DECISION.md`.
