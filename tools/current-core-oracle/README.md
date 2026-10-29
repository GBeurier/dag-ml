# Current Core archive conformance process

This standalone, non-publishable tool qualifies N4MM format 3 through the
actual Core 0.4.6 archive writer, typed loader and member accessors. It owns no
archive parser, model algorithm or training code. Input and output are bounded
to 16 MiB, generated inputs stay in a private temporary directory, and the
real reader must reject a flipped Package V2 member.

The historical Core 0.4.3 tool remains separate and exact. The native witness
selects this current reader explicitly for format 3, retaining ABI minimum
2.20; it never relabels format 3 as a historical format or drops that minimum.

After Core 0.4.6 is published, build with:

```sh
cargo build --manifest-path tools/current-core-oracle/Cargo.toml
export DAG_ML_CURRENT_CORE_ORACLE="$PWD/tools/current-core-oracle/target/debug/dag-ml-current-core-oracle"
```

Before publication, build using an explicit local Cargo configuration that
patches the candidate Core and its candidate dependencies to their owner
source checkouts. Keep the source SHAs, dirty-source hashes if any, resolved
manifests/locks and binary SHA-256 in the local qualification receipt. Such a
build proves a local source cohort, not registry availability. The native
witness refuses to substitute the historical reader if this one is absent.
