# Working strategy — migrate the core while keeping prod alive

**ARCHIVÉ — 17 septembre 2026.** Stratégie de migration des versions 0.x ; ne constitue plus une instruction de branchement ou de bascule.

Le document historique complet est conservé dans les archives de travail du
mainteneur, qui ne font pas partie de la distribution publique.

Pour reprendre le travail :

- [Support actuel](../SUPPORTED.md).
- [Contrats actuels](../COORDINATOR_SPEC.md).
- `BACKLOG_ECOSYSTEME.md` (pilotage mainteneur hors dépôt).
- `ROADMAP_CONSOLIDATION_MULTIMODALE.md` (pilotage mainteneur hors dépôt).

Les anciennes priorités et commandes ne sont plus un plan à exécuter.

<details>
<summary>Anciennes sections de l’archive</summary>

- Constraints this must respect.
- Recommendation — *not* a fork, *not* a new repo: **selector-on-main + integration branch + worktree**.
- Destination = dag-ml directly, consumed as an optional dependency.
- Integration on main behind the selector, in flag-gated increments.
- A long-lived branch + worktree only for non-isolatable spikes.
- Local setup, remote release, alternatives, and first concrete steps.

</details>
