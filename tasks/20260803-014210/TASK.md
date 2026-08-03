# Examples cannot start the app from a running event loop

- PRIORITY: 0
- TAGS: bug,backlog,examples,db
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

As a Scufris contributor, I want the runnable examples to start the app from
inside an async entry point, so that the examples still demonstrate what they
document.

## Notes

- Found by review round 1 of 20260803-002141, on master rather than on that
  branch. Not that branch's regression; filed rather than held against it.
- Since 18b117b, `create_app` opens a transaction during startup
  (`sessions.prune`, `scufris/app.py:1107`), and `Database.transaction()`
  refuses a thread with a running event loop. Any example that calls
  `create_app` from inside `asyncio.run` therefore dies at startup.
- Reproduced: `python examples/comms_loop.py` exits 1 with "a transaction
  cannot be opened on a thread with a running event loop".
  `examples/telegram_approval.py:125` and `examples/telegram_bot.py:134` have
  the same shape and want the same check.
- The fix is a choice between building the app before entering the loop and
  offloading the synchronous startup block, and that choice belongs to
  planning, not to this record.
- Worth a check that keeps it fixed: the examples are currently only proven by
  being run by hand.
