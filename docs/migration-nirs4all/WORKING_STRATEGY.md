# Working strategy — migrate the core while keeping prod alive

**ARCHIVÉ — 17 septembre 2026.** Stratégie de migration des versions 0.x ; ne constitue plus une instruction de branchement ou de bascule.

[Lire le document historique complet](../../archives/planification/2026-09-17/WORKING_STRATEGY.md).

Pour reprendre le travail :

- [Support actuel](../SUPPORTED.md).
- [Contrats actuels](../COORDINATOR_SPEC.md).
- [Backlog actif](../../../BACKLOG_ECOSYSTEME.md).
- [Premier chantier](../../../ROADMAP_CONSOLIDATION_MULTIMODALE.md).

Les anciennes priorités et commandes ne sont plus un plan à exécuter.

<details>
<summary>Anciennes sections — liens vers l’archive</summary>

- <a id="constraints-this-must-respect"></a>[Constraints this must respect](../../archives/planification/2026-09-17/WORKING_STRATEGY.md#constraints-this-must-respect)
- <a id="recommendation--not-a-fork-not-a-new-repo-selector-on-main--integration-branch--worktree"></a>[Recommendation — *not* a fork, *not* a new repo: **selector-on-main + integration branch + worktree**](../../archives/planification/2026-09-17/WORKING_STRATEGY.md#recommendation--not-a-fork-not-a-new-repo-selector-on-main--integration-branch--worktree)
- <a id="1-destination--dag-ml-directly-recommended-consumed-as-an-optional-dependency"></a>[1. Destination = dag-ml directly (recommended), consumed as an optional dependency](../../archives/planification/2026-09-17/WORKING_STRATEGY.md#1-destination--dag-ml-directly-recommended-consumed-as-an-optional-dependency)
- <a id="2-integration-lives-on-main-behind-the-selector-in-flag-gated-increments"></a>[2. Integration lives on main behind the selector, in flag-gated increments](../../archives/planification/2026-09-17/WORKING_STRATEGY.md#2-integration-lives-on-main-behind-the-selector-in-flag-gated-increments)
- <a id="3-a-long-lived-branch--worktree-only-for-the-churny-not-yet-flag-isolatable-spikes"></a>[3. A long-lived branch + worktree **only** for the churny, not-yet-flag-isolatable spikes](../../archives/planification/2026-09-17/WORKING_STRATEGY.md#3-a-long-lived-branch--worktree-only-for-the-churny-not-yet-flag-isolatable-spikes)
- <a id="local-setup--maintain-prod-and-migrate-at-the-same-time-zero-context-switch"></a>[Local setup — maintain prod and migrate at the same time, zero context-switch](../../archives/planification/2026-09-17/WORKING_STRATEGY.md#local-setup--maintain-prod-and-migrate-at-the-same-time-zero-context-switch)
- <a id="remote--release"></a>[Remote & release](../../archives/planification/2026-09-17/WORKING_STRATEGY.md#remote--release)
- <a id="why-not-the-alternatives"></a>[Why not the alternatives](../../archives/planification/2026-09-17/WORKING_STRATEGY.md#why-not-the-alternatives)
- <a id="first-concrete-steps-once-decisions-12-are-set"></a>[First concrete steps (once decisions #1/#2 are set)](../../archives/planification/2026-09-17/WORKING_STRATEGY.md#first-concrete-steps-once-decisions-12-are-set)

</details>
