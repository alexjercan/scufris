# Add read-only email search and calendar agenda plugins

- STATUS: OPEN
- PRIORITY: 0
- TAGS: feature,backlog,plugins,email,calendar

## Story

As a personal-assistant user, I want narrowly scoped email search/drafting and
calendar agenda access, so that specialized agents can prepare communications
and briefings without being able to send mail or alter my calendar by default.

## Steps

- [ ] Define provider-neutral email search/message/thread/draft and calendar
      agenda/event models with stable external references and provenance.
- [ ] Implement sample out-of-process plugins for one configured email provider
      and one calendar provider using secret references and least-privilege
      read scopes.
- [ ] Expose search, read thread, agenda, and event detail tools plus local
      email-draft artifacts; do not implement send/create/update/delete.
- [ ] Add HTML-email sanitization, attachment metadata, pagination, timezone,
      recurrence, all-day, cancellation, and provider rate-limit handling.
- [ ] Provide agent templates for inbox briefing, meeting preparation, and
      email drafting with visible capability grants.
- [ ] Add stub-provider integration tests and an opt-in live contract test that
      cannot mutate provider data.
- [ ] Add a runnable stubbed example that produces an inbox briefing, meeting
      agenda, and local email-draft artifact without provider writes.
- [ ] Document the separate future approval work required for send or calendar
      writes.

## Definition of Done

- Stubbed plugins search/read email and calendar data end to end through an
  approved preset-derived agent
  (test: `test_personal_information_plugins_end_to_end`).
- Email HTML and attachments cannot execute active content or escape artifact
  boundaries (test: `test_email_content_is_sanitized`).
- Timezones, recurrence, pagination, and rate limits retain correct semantics
  (test: `test_calendar_and_email_edge_cases`).
- No plugin tool exposes send/create/update/delete capability
  (test: `test_personal_information_plugins_are_read_only`).
- The stubbed personal-information example succeeds
  (cmd: `python examples/personal_information_plugins.py`).
- manual: a real inbox briefing and meeting agenda are useful without exposing
  write access.

## Notes

- Epic: 20260729-102210.
- Depends on: 20260729-102207, 20260729-102208, 20260729-102919, and
  20260729-102212.
- Sending email or changing calendar state must be planned as separate
  capability-gated tasks after read-only use is proven.

## Flow State

- FLOW STEP: PLANNING
