# Retro: the NixOS configuration change flow (R3)

- TASK: 20260729-125035
- CLOSED with 1 review round (7 findings: 4 MAJOR fixed, 3 MINOR - 1 fixed,
  2 accepted with reasons)

## What went well

- **The operator's re-frame arrived before any code existed.** The plan gate
  asked which of three EDIT shapes to build; the answer was "none of them -
  `nix.dotfiles` is just a project". That deleted five planned steps, a typed-edit
  module and its anchor-finding, and left a task that is only the part which
  genuinely needs the machine. A gate that changes the shape of the work is the
  gate doing its job.
- **Measuring before writing paid for itself immediately.** Reading this host's
  `nixos-rebuild` gave the exact two-command activation, including the
  `systemd-run --unit=nixos-rebuild-switch-to-configuration` wrapper - which
  turned out to matter twice: it survives an activation that restarts scufris,
  and its shared unit name is a free mutex against a hand-run `nixos-rebuild`.
  None of that would have been invented from first principles.
- **The VM test earned its cost in one cycle.** Both findings it produced
  (`nix-command` not being enabled; a test VM having no system profile) were
  reachable by reading and were not read. It also proved the split-state path by
  accident - the rollback's bootloader install failed and the record said "step 2
  of 2 failed after 1 succeeded... THIS boot still runs the old one while the
  NEXT boot would run the new one" - before any test asserted it.
- **Sabotaging each new guard.** Three fixes, three tests, each watched red with
  the fix neutered and green with it restored, on a committed tree so the restore
  could not eat the fix.

## What went wrong

- **The plan asked the wrong question first.** It offered three artifact shapes
  for the EDIT (typed verbs, wider typed verbs, free-form) when the real fork was
  ownership: does Scufris own the edit at all? The signal was already in the
  task's own Story - "add this package", "open this port", "turn on that service"
  is three narrow surfaces, i.e. an editor - and in the actor-aware orchestrator
  spike, whose Projects model already owns worktrees, commits and reviews. Both
  were read during planning and neither prompted the question.
- **A preview that executes the thing it previews got as far as the code.** It
  was in the approved plan and in the spike's own table. It came out during
  implementation, not review, but it should not have survived the plan: "run the
  proposed configuration's own binary, as root, before the approval" contradicts
  the framework's first sentence.
- **The attribute probe was put where it was convenient to write, not where it
  costs nothing.** A synchronous flake evaluation in a request whose only caller
  has a 15-second timeout. The timeout is one grep away in the same repository.
- **The repository path was left caller-controlled with a note in the DECISION
  saying an allowlist "wouldn't prevent malicious configs".** True and irrelevant:
  the point of pinning it is not to stop bad Nix, it is that "the server built
  what it activates" means nothing if the caller picks which repository the server
  reads. Arguing why a control is imperfect is not a reason to skip it.

## Lessons

1. `ask-who-owns-it-before-asking-what-shape-it-is`: when a feature touches a
   repo, a project or a surface that already has an owner elsewhere in the
   system, the first planning question is "whose job is this", not "which of my
   three designs". Three artifact options for something you should not be
   building at all is a well-formed question with no right answer.
2. `a-preview-must-not-execute-what-it-previews`: obtaining a better preview by
   running the unapproved artifact's own code (as root, before the approval)
   trades the whole approval for a nicer panel. Where the only honest preview is
   narrower, ship the narrow one and say what is missing and why.
3. `read-the-caller-s-timeout-before-putting-work-in-a-request`: a synchronous
   probe is only free if every caller can wait. The MCP tools' API timeout is
   15s; a flake evaluation is seconds-to-minutes. Long work belongs in the run
   the request starts, where failure lands on a record instead of on a socket.
4. `pin-the-input-a-caller-should-not-choose`: "which revision" is the caller's
   business, "which repository" is not. When a security property reads "the
   server builds what it activates", check every input to that build for who
   supplies it - a caller-supplied repository defeats it as surely as a
   caller-supplied store path.
5. `run-the-environment-test-first-not-last`: the VM test was the last step in
   the plan and the first thing to find a real bug, twice. For a feature whose
   whole point is a privileged operation, the environment proof belongs early -
   it is where the assumptions about the environment live.
