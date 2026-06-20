"""Generate labeled synthetic syscall traces for ABI tooling and detector benchmarks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

try:
    from .defaults import (
        DEFAULT_INBOX_ROOT,
        DEFAULT_ORGANIZED_ROOT,
        DEFAULT_SYNTHETIC_SEED,
        default_manifest_path,
    )
    from .tables import load_syscall_abi_mapping
except ImportError:
    from defaults import (  # type: ignore
        DEFAULT_INBOX_ROOT,
        DEFAULT_ORGANIZED_ROOT,
        DEFAULT_SYNTHETIC_SEED,
        default_manifest_path,
    )
    from tables import load_syscall_abi_mapping  # type: ignore


FOLDER_TO_ABI = {
 "i386": "i386",
 "x86_64": "x86_64",
 "arm": "arm",
 "arm64": "arm64_asm_generic",
}

BENIGN_CANDIDATES = (
 "read",
 "write",
 "open",
 "openat",
 "close",
 "stat",
 "fstat",
 "lstat",
 "newfstatat",
 "fstatat64",
 "lseek",
 "mmap",
 "mmap2",
 "munmap",
 "brk",
 "access",
 "getpid",
 "getuid",
 "gettimeofday",
 "clock_gettime",
 "clock_getres",
 "futex",
 "poll",
 "ppoll",
 "select",
 "newselect",
 "ioctl",
 "readlink",
 "readlinkat",
 "getdents",
 "getdents64",
 "exit",
 "exit_group",
 "rt_sigaction",
 "rt_sigprocmask",
 "nanosleep",
 "pipe",
 "pipe2",
 "dup",
 "dup2",
 "dup3",
)

SUSPICIOUS_CANDIDATES = (
 "execve",
 "socket",
 "socketcall",
 "connect",
 "accept",
 "accept4",
 "sendto",
 "sendmsg",
 "recvfrom",
 "recvmsg",
 "bind",
 "listen",
 "shutdown",
 "chmod",
 "unlink",
 "rename",
 "mkdir",
 "rmdir",
 "link",
 "symlink",
 "setuid",
 "setgid",
 "setreuid",
 "setresuid",
 "setresgid",
 "ptrace",
 "mount",
 "kill",
 "clone",
 "fork",
 "vfork",
 "mprotect",
 "prlimit64",
 "process_vm_readv",
 "process_vm_writev",
 "setns",
 "unshare",
 "chroot",
)

AGGRESSIVE_MOTIFS = (
    ("setuid", "setgid", "setreuid", "setresuid", "setresgid", "execve"),
    ("ptrace", "mprotect", "clone", "execve", "kill"),
    ("unshare", "chroot", "mount", "execve", "kill"),
    ("mkdir", "chmod", "rename", "unlink", "symlink"),
    ("fork", "clone", "execve", "mprotect", "kill"),
    ("chmod", "unlink", "rename", "link", "execve"),
)

BENIGN_MOTIFS = (
    ("openat", "fstat", "read", "close"),
    ("open", "read", "read", "close"),
    ("getpid", "clock_gettime", "futex", "nanosleep"),
    ("poll", "read", "write", "close"),
    ("pipe2", "dup", "dup2", "close"),
    ("rt_sigaction", "rt_sigprocmask", "gettimeofday", "clock_getres"),
)

BENIGN_CONTROL_FILE_CANDIDATES = (
 "read",
 "open",
 "openat",
 "close",
 "stat",
 "fstat",
 "lstat",
 "newfstatat",
 "fstatat64",
 "lseek",
 "access",
 "readlink",
 "readlinkat",
 "getdents",
 "getdents64",
)

BENIGN_CONTROL_PROCESS_CANDIDATES = (
 "getpid",
 "gettimeofday",
 "clock_getres",
 "rt_sigaction",
 "rt_sigprocmask",
 "exit",
 "exit_group",
)

BENIGN_CONTROL_MEMORY_CANDIDATES = (
 "brk",
 "mmap",
 "mmap2",
 "munmap",
)

BENIGN_CONTROL_MOTIFS = (
    ("openat", "fstat", "read", "close"),
    ("open", "read", "close"),
    ("rt_sigaction", "rt_sigprocmask", "gettimeofday", "clock_getres"),
    ("stat", "lstat", "readlinkat"),
    ("getdents64", "lseek", "close"),
    ("brk", "mmap2", "munmap"),
)

MALWARE_TARGET_PRIMARY_SUSPICIOUS = (
 "execve",
 "kill",
 "clone",
 "mprotect",
 "ptrace",
 "fork",
 "chmod",
 "unlink",
 "rename",
 "link",
 "mkdir",
 "symlink",
)

MALWARE_TARGET_SUPPORT_CANDIDATES = (
 "pipe",
 "pipe2",
 "dup2",
 "close",
 "getuid",
 "readlink",
 "readlinkat",
 "access",
 "munmap",
 "gettimeofday",
 "clock_getres",
 "ppoll",
 "lseek",
)

MALWARE_TARGET_PRESETS = {
 "exec_kill_chain": {
 "primary_candidates": (
 "fork",
 "clone",
 "execve",
 "mprotect",
 "kill",
        ),
 "motifs": (
            ("fork", "clone", "execve", "mprotect", "kill"),
            ("clone", "execve", "mprotect", "kill"),
            ("fork", "execve", "kill"),
        ),
    },
 "file_rewrite_exec": {
 "primary_candidates": (
 "chmod",
 "unlink",
 "rename",
 "link",
 "execve",
        ),
 "motifs": (
            ("chmod", "unlink", "rename", "link", "execve"),
            ("unlink", "rename", "execve"),
            ("chmod", "rename", "execve"),
        ),
    },
 "path_link_pivot": {
 "primary_candidates": (
 "mkdir",
 "chmod",
 "rename",
 "unlink",
 "symlink",
 "link",
 "execve",
        ),
 "support_candidates": (
 "pipe",
 "dup2",
 "ppoll",
 "gettimeofday",
 "readlink",
        ),
 "motifs": (
            ("mkdir", "link", "rename", "unlink", "execve"),
            ("mkdir", "chmod", "rename", "unlink", "execve"),
            ("mkdir", "link", "rename", "unlink", "symlink", "execve"),
            ("mkdir", "unlink", "rename", "link", "execve"),
        ),
 "motif_target_ratio": 0.78,
 "min_motif_tokens": 26,
 "support_target_min": 4,
 "bridge_support_prob": 0.16,
    },
 "ptrace_loader_chain": {
 "primary_candidates": (
 "ptrace",
 "mprotect",
 "clone",
 "execve",
 "kill",
        ),
 "motifs": (
            ("ptrace", "mprotect", "clone", "execve", "kill"),
            ("ptrace", "clone", "execve", "kill"),
            ("ptrace", "mprotect", "execve"),
        ),
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
 "Generate synthetic syscall-name traces in inbox/<abi>/ with a benign "
 "and malware-like mix for batch conversion testing."
        )
    )
    parser.add_argument(
 "--inbox-root",
        default=DEFAULT_INBOX_ROOT.as_posix(),
        help="Root inbox directory that contains the ABI subfolders.",
    )
    parser.add_argument(
 "--count-per-abi",
        type=int,
        default=100,
        help="How many synthetic files to generate in each ABI subfolder.",
    )
    parser.add_argument(
 "--min-tokens",
        type=int,
        default=20,
        help="Minimum number of syscall names per generated file.",
    )
    parser.add_argument(
 "--max-tokens",
        type=int,
        default=32,
        help="Maximum number of syscall names per generated file.",
    )
    parser.add_argument(
 "--seed",
        type=int,
        default=DEFAULT_SYNTHETIC_SEED,
        help="Deterministic random seed.",
    )
    parser.add_argument(
 "--prefix",
        default="synthetic_mixed",
        help="Filename prefix used for generated files.",
    )
    parser.add_argument(
 "--profile",
        choices=("mixed", "aggressive"),
        default="mixed",
        help="Trace composition profile. 'aggressive' biases toward clustered suspicious syscall motifs.",
    )
    parser.add_argument(
 "--compat-abi",
        default=None,
        help="Optional ABI name to use as a compatibility filter for syscall names.",
    )
    parser.add_argument(
 "--compat-max-number",
        type=int,
        default=None,
        help="Optional maximum syscall number allowed in the compatibility ABI.",
    )
    parser.add_argument(
 "--abis",
        nargs="+",
        choices=tuple(FOLDER_TO_ABI),
        default=None,
        help="Optional ABI inbox subfolders to generate. Defaults to all.",
    )
    parser.add_argument(
 "--label-mode",
        choices=("none", "binary_controls"),
        default="none",
        help=(
 "Optional ground-truth labeling mode. 'binary_controls' splits each ABI batch into "
 "'benign_control' and 'malware_targeted' samples and writes a manifest JSON."
        ),
    )
    parser.add_argument(
 "--malware-ratio",
        type=float,
        default=0.5,
        help=(
 "Fraction of samples per ABI that should be labeled malware_targeted when "
 "--label-mode=binary_controls."
        ),
    )
    parser.add_argument(
 "--malware-target-presets",
        nargs="+",
        choices=tuple(MALWARE_TARGET_PRESETS),
        default=None,
        help=(
 "Optional malware_targeted motif presets to use when --label-mode=binary_controls. "
 "Defaults to all presets with an even split across malware_targeted samples."
        ),
    )
    parser.add_argument(
 "--manifest-out",
        default=None,
        help=(
 "Optional output path for the generation manifest JSON. Defaults to "
 "inbox/<prefix>.manifest.json when labels are enabled."
        ),
    )
    return parser


def filter_available_names(mapping_names: set[str], candidates: tuple[str, ...]) -> list[str]:
    return [name for name in candidates if name in mapping_names]


def filter_compatible_names(
    names: list[str],
    compat_mapping,
    compat_max_number: int | None,
) -> list[str]:
    if compat_mapping is None:
        return list(names)

    compatible: list[str] = []
    for name in names:
        numbers = compat_mapping.name_to_numbers.get(name, ())
        if not numbers:
            continue
        if compat_max_number is not None and not any(int(number) <= compat_max_number for number in numbers):
            continue
        compatible.append(name)
    return compatible


def clean_existing_generated_files(folder: Path, prefix: str) -> None:
    for path in folder.glob(f"{prefix}_*.txt"):
        path.unlink()


def choose_tokens(pool: list[str], count: int, rng: random.Random) -> list[str]:
    if not pool or count <= 0:
        return []
    return [rng.choice(pool) for _ in range(count)]


def build_trace_tokens(
    benign_pool: list[str],
    suspicious_pool: list[str],
    extra_pool: list[str],
    token_count: int,
    rng: random.Random,
) -> list[str]:
    suspicious_count = max(6, token_count // 3)
    benign_count = max(10, token_count // 2)
    if suspicious_count + benign_count > token_count:
        benign_count = max(8, token_count - suspicious_count)
    extra_count = max(0, token_count - suspicious_count - benign_count)

    tokens = []
    tokens.extend(choose_tokens(benign_pool, benign_count, rng))
    tokens.extend(choose_tokens(suspicious_pool, suspicious_count, rng))
    tokens.extend(choose_tokens(extra_pool, extra_count, rng))
    rng.shuffle(tokens)
    return tokens


def build_motif_pool(pool: list[str], motif_library: tuple[tuple[str, ...], ...]) -> list[tuple[str, ...]]:
    pool_set = set(pool)
    motifs: list[tuple[str, ...]] = []
    for motif in motif_library:
        filtered = tuple(name for name in motif if name in pool_set)
        if len(filtered) >= 3:
            motifs.append(filtered)
    return motifs


def build_aggressive_motif_pool(suspicious_pool: list[str]) -> list[tuple[str, ...]]:
    return build_motif_pool(suspicious_pool, AGGRESSIVE_MOTIFS)


def build_trace_tokens_benign_control(
    file_pool: list[str],
    process_pool: list[str],
    memory_pool: list[str],
    token_count: int,
    rng: random.Random,
) -> list[str]:
    combined_pool = list(dict.fromkeys(file_pool + process_pool + memory_pool))
    motifs = build_motif_pool(combined_pool, BENIGN_CONTROL_MOTIFS)
    file_target = max(18, round(token_count * 0.58))
    process_target = max(8, round(token_count * 0.24))
    memory_target = max(4, token_count - file_target - process_target)
    if file_target + process_target + memory_target > token_count:
        memory_target = max(2, token_count - file_target - process_target)

    tokens: list[str] = []
    tokens.extend(choose_tokens(file_pool, file_target, rng))
    tokens.extend(choose_tokens(process_pool, process_target, rng))
    tokens.extend(choose_tokens(memory_pool, memory_target, rng))

    if motifs:
        motif_inserts = rng.randint(0, 2)
        for _ in range(motif_inserts):
            motif = list(rng.choice(motifs))
            if rng.random() < 0.2:
                motif = list(reversed(motif))
            tokens.extend(motif)

    while len(tokens) < token_count:
        burst_pool = rng.choice([file_pool, file_pool, process_pool, memory_pool])
        tokens.extend(choose_tokens(burst_pool, 1, rng))

    rng.shuffle(tokens)
    return tokens[:token_count]


def resolve_manifest_path(args: argparse.Namespace) -> Path | None:
    if args.label_mode == "none":
        return None
    if args.manifest_out:
        return Path(args.manifest_out)
    return default_manifest_path(str(args.prefix))


def build_label_plan(
    count_per_abi: int,
    label_mode: str,
    malware_ratio: float,
) -> list[str | None]:
    if label_mode == "none":
        return [None] * int(count_per_abi)

    total = int(count_per_abi)
    if total <= 0:
        return []
    malware_count = int(round(total * float(malware_ratio)))
    malware_count = min(total, max(0, malware_count))
    benign_count = total - malware_count
    return (["benign_control"] * benign_count) + (["malware_targeted"] * malware_count)


def build_malware_target_preset_plan(
    malware_count: int,
    preset_names: list[str],
    rng: random.Random,
) -> list[str]:
    if malware_count <= 0:
        return []
    if not preset_names:
        raise ValueError("At least one malware target preset is required when malware_targeted samples are enabled.")

    plan = [preset_names[index % len(preset_names)] for index in range(int(malware_count))]
    rng.shuffle(plan)
    return plan


def build_trace_tokens_aggressive(
    benign_pool: list[str],
    suspicious_pool: list[str],
    token_count: int,
    rng: random.Random,
) -> list[str]:
    suspicious_target = min(token_count, max(16, round(token_count * 0.72)))
    benign_remaining = max(0, token_count - suspicious_target)
    motifs = build_aggressive_motif_pool(suspicious_pool)

    tokens: list[str] = []
    suspicious_added = 0
    while suspicious_added < suspicious_target:
        if motifs and rng.random() < 0.85:
            motif = list(rng.choice(motifs))
            if rng.random() < 0.35:
                motif = list(reversed(motif))
            block = motif[: max(1, min(len(motif), suspicious_target - suspicious_added))]
        else:
            block = [rng.choice(suspicious_pool)]

        tokens.extend(block)
        suspicious_added += len(block)

        if benign_remaining > 0 and len(tokens) < token_count and rng.random() < 0.45:
            benign_burst = min(benign_remaining, rng.randint(1, 3))
            tokens.extend(choose_tokens(benign_pool, benign_burst, rng))
            benign_remaining -= benign_burst

    if benign_remaining > 0:
        tokens.extend(choose_tokens(benign_pool, benign_remaining, rng))

    while len(tokens) < token_count:
        tokens.append(rng.choice(suspicious_pool))
    return tokens[:token_count]


def build_trace_tokens_malware_targeted(
    primary_suspicious_pool: list[str],
    support_pool: list[str],
    motif_pool: list[tuple[str, ...]],
    token_count: int,
    rng: random.Random,
    motif_target_ratio: float = 0.72,
    min_motif_tokens: int = 24,
    support_target_min: int = 6,
    bridge_support_prob: float = 0.25,
) -> list[str]:
    if not motif_pool:
        return build_trace_tokens_aggressive(
            benign_pool=support_pool or primary_suspicious_pool,
            suspicious_pool=primary_suspicious_pool,
            token_count=token_count,
            rng=rng,
        )

    motif_token_target = min(token_count, max(int(min_motif_tokens), round(token_count * float(motif_target_ratio))))
    support_target = max(int(support_target_min), token_count - motif_token_target)

    tokens: list[str] = []
    while len(tokens) < motif_token_target:
        motif = list(rng.choice(motif_pool))
        if len(tokens) + len(motif) > motif_token_target:
            motif = motif[: motif_token_target - len(tokens)]
        tokens.extend(motif)

        if support_pool and len(tokens) < motif_token_target and rng.random() < float(bridge_support_prob):
            # Insert a single support token between motif runs.
            tokens.extend(choose_tokens(support_pool, 1, rng))

    if support_pool:
        tokens.extend(choose_tokens(support_pool, support_target, rng))

    while len(tokens) < token_count:
        refill_pool = primary_suspicious_pool if rng.random() < 0.85 else support_pool or primary_suspicious_pool
        tokens.extend(choose_tokens(refill_pool, 1, rng))

    return tokens[:token_count]


def main() -> None:
    args = build_parser().parse_args()
    if args.count_per_abi <= 0:
        raise ValueError("--count-per-abi must be > 0")
    if args.min_tokens <= 0 or args.max_tokens <= 0:
        raise ValueError("--min-tokens and --max-tokens must be > 0")
    if args.min_tokens > args.max_tokens:
        raise ValueError("--min-tokens cannot be greater than --max-tokens")
    if not 0.0 <= float(args.malware_ratio) <= 1.0:
        raise ValueError("--malware-ratio must be between 0.0 and 1.0")

    project_root = Path(__file__).resolve().parents[2]
    inbox_root = (project_root / args.inbox_root).resolve()
    inbox_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))
    manifest_path = resolve_manifest_path(args)
    compat_mapping = None
    selected_malware_target_presets = list(args.malware_target_presets or MALWARE_TARGET_PRESETS.keys())
    if args.compat_abi:
        compat_mapping = load_syscall_abi_mapping(str(args.compat_abi), project_root=project_root)

    generated_total = 0
    manifest_entries: list[dict[str, object]] = []
    per_abi_summary: dict[str, dict[str, object]] = {}
    selected_folders = (
        {folder_name: FOLDER_TO_ABI[folder_name] for folder_name in args.abis}
        if args.abis
        else dict(FOLDER_TO_ABI)
    )

    for folder_name, abi in selected_folders.items():
        target_dir = inbox_root / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        clean_existing_generated_files(target_dir, str(args.prefix))

        mapping = load_syscall_abi_mapping(abi, project_root=project_root)
        all_names = set(mapping.name_to_numbers)
        benign_pool = filter_available_names(all_names, BENIGN_CANDIDATES)
        suspicious_pool = filter_available_names(all_names, SUSPICIOUS_CANDIDATES)
        extra_pool = sorted(all_names - set(benign_pool) - set(suspicious_pool))
        benign_control_file_pool = filter_available_names(all_names, BENIGN_CONTROL_FILE_CANDIDATES)
        benign_control_process_pool = filter_available_names(all_names, BENIGN_CONTROL_PROCESS_CANDIDATES)
        benign_control_memory_pool = filter_available_names(all_names, BENIGN_CONTROL_MEMORY_CANDIDATES)
        malware_target_support_pool = filter_available_names(all_names, MALWARE_TARGET_SUPPORT_CANDIDATES)
        benign_pool = filter_compatible_names(benign_pool, compat_mapping, args.compat_max_number)
        suspicious_pool = filter_compatible_names(suspicious_pool, compat_mapping, args.compat_max_number)
        extra_pool = filter_compatible_names(extra_pool, compat_mapping, args.compat_max_number)
        benign_control_file_pool = filter_compatible_names(
            benign_control_file_pool,
            compat_mapping,
            args.compat_max_number,
        )
        benign_control_process_pool = filter_compatible_names(
            benign_control_process_pool,
            compat_mapping,
            args.compat_max_number,
        )
        benign_control_memory_pool = filter_compatible_names(
            benign_control_memory_pool,
            compat_mapping,
            args.compat_max_number,
        )
        malware_target_support_pool = filter_compatible_names(
            malware_target_support_pool,
            compat_mapping,
            args.compat_max_number,
        )
        malware_target_preset_configs: dict[str, dict[str, object]] = {}
        for preset_name in selected_malware_target_presets:
            preset_definition = MALWARE_TARGET_PRESETS[preset_name]
            primary_pool = filter_available_names(all_names, preset_definition["primary_candidates"])
            primary_pool = filter_compatible_names(primary_pool, compat_mapping, args.compat_max_number)
            support_candidates = tuple(preset_definition.get("support_candidates", MALWARE_TARGET_SUPPORT_CANDIDATES))
            support_pool = filter_available_names(all_names, support_candidates)
            support_pool = filter_compatible_names(support_pool, compat_mapping, args.compat_max_number)
            motif_pool = build_motif_pool(primary_pool, preset_definition["motifs"])
            if len(primary_pool) < 3:
                raise ValueError(
 f"Not enough malware-target primary syscall names available for {abi} preset {preset_name}: "
 f"{len(primary_pool)}"
                )
            if not motif_pool:
                raise ValueError(f"No malware-target motifs available for {abi} preset {preset_name}.")
            malware_target_preset_configs[preset_name] = {
 "primary_pool": primary_pool,
 "support_pool": support_pool or malware_target_support_pool,
 "motif_pool": motif_pool,
 "motif_target_ratio": float(preset_definition.get("motif_target_ratio", 0.72)),
 "min_motif_tokens": int(preset_definition.get("min_motif_tokens", 24)),
 "support_target_min": int(preset_definition.get("support_target_min", 6)),
 "bridge_support_prob": float(preset_definition.get("bridge_support_prob", 0.25)),
            }

        if len(benign_pool) < 10:
            raise ValueError(f"Not enough benign syscall names available for {abi}: {len(benign_pool)}")
        if len(suspicious_pool) < 6:
            raise ValueError(f"Not enough suspicious syscall names available for {abi}: {len(suspicious_pool)}")
        if len(benign_control_file_pool) < 8:
            raise ValueError(
 f"Not enough benign-control file syscall names available for {abi}: {len(benign_control_file_pool)}"
            )
        if len(benign_control_process_pool) < 4:
            raise ValueError(
 f"Not enough benign-control process syscall names available for {abi}: {len(benign_control_process_pool)}"
            )
        if len(benign_control_memory_pool) < 2:
            raise ValueError(
 f"Not enough benign-control memory syscall names available for {abi}: {len(benign_control_memory_pool)}"
            )

        label_plan = build_label_plan(
            count_per_abi=int(args.count_per_abi),
            label_mode=str(args.label_mode),
            malware_ratio=float(args.malware_ratio),
        )
        rng.shuffle(label_plan)
        malware_target_count = int(sum(1 for label in label_plan if label == "malware_targeted"))
        malware_target_preset_plan = build_malware_target_preset_plan(
            malware_count=malware_target_count,
            preset_names=selected_malware_target_presets,
            rng=rng,
        )
        label_counts = {
 "benign_control": int(sum(1 for label in label_plan if label == "benign_control")),
 "malware_targeted": malware_target_count,
        }
        malware_target_preset_counts = {
            preset_name: int(sum(1 for preset in malware_target_preset_plan if preset == preset_name))
            for preset_name in selected_malware_target_presets
        }
        per_abi_summary[folder_name] = {
 "abi": abi,
 "files_total": int(args.count_per_abi),
 "label_counts": label_counts,
 "malware_target_preset_counts": malware_target_preset_counts,
        }
        label_indices = {
 "benign_control": 0,
 "malware_targeted": 0,
        }
        malware_target_preset_indices = {preset_name: 0 for preset_name in selected_malware_target_presets}

        for sample_index, label in enumerate(label_plan, start=1):
            token_count = rng.randint(int(args.min_tokens), int(args.max_tokens))
            target_preset = None
            if label == "benign_control":
                tokens = build_trace_tokens_benign_control(
                    file_pool=benign_control_file_pool,
                    process_pool=benign_control_process_pool,
                    memory_pool=benign_control_memory_pool,
                    token_count=token_count,
                    rng=rng,
                )
            elif label == "malware_targeted":
                target_preset = malware_target_preset_plan[label_indices["malware_targeted"]]
                preset_config = malware_target_preset_configs[target_preset]
                tokens = build_trace_tokens_malware_targeted(
                    primary_suspicious_pool=list(preset_config["primary_pool"]),
                    support_pool=list(preset_config["support_pool"]),
                    motif_pool=list(preset_config["motif_pool"]),
                    token_count=token_count,
                    rng=rng,
                    motif_target_ratio=float(preset_config["motif_target_ratio"]),
                    min_motif_tokens=int(preset_config["min_motif_tokens"]),
                    support_target_min=int(preset_config["support_target_min"]),
                    bridge_support_prob=float(preset_config["bridge_support_prob"]),
                )
            elif args.profile == "aggressive":
                tokens = build_trace_tokens_aggressive(
                    benign_pool=benign_pool,
                    suspicious_pool=suspicious_pool,
                    token_count=token_count,
                    rng=rng,
                )
            else:
                tokens = build_trace_tokens(
                    benign_pool=benign_pool,
                    suspicious_pool=suspicious_pool,
                    extra_pool=extra_pool,
                    token_count=token_count,
                    rng=rng,
                )
            if label is None:
                file_name = f"{args.prefix}_{sample_index:03d}.txt"
                sample_id = Path(file_name).stem
                intended_label = None
            else:
                label_indices[label] += 1
                if label == "malware_targeted" and target_preset is not None:
                    malware_target_preset_indices[target_preset] += 1
                    file_name = (
 f"{args.prefix}_{label}_{target_preset}_{malware_target_preset_indices[target_preset]:03d}.txt"
                    )
                else:
                    file_name = f"{args.prefix}_{label}_{label_indices[label]:03d}.txt"
                sample_id = Path(file_name).stem
                intended_label = label

            file_path = target_dir / file_name
            file_path.write_text(" ".join(tokens) + "\n", encoding="utf-8")
            generated_total += 1
            if manifest_path is not None:
                relative_source = file_path.relative_to(project_root)
                expected_converted = (
                    DEFAULT_ORGANIZED_ROOT
                    / folder_name
                    / "converted"
                    / f"{sample_id}.name_to_number.txt"
                )
                manifest_entries.append(
                    {
 "sample_id": sample_id,
 "abi_folder": folder_name,
 "abi": abi,
 "intended_label": intended_label,
 "target_preset": target_preset,
 "profile": "benign_control" if intended_label == "benign_control" else args.profile,
                        "token_count": token_count,
                        "source_file": str(relative_source),
                        "expected_converted_file": str(expected_converted),
                    }
                )

        print(
            f"OK {folder_name}: generated {int(args.count_per_abi)} files "
            f"with {len(benign_pool)} benign names and {len(suspicious_pool)} malware-like names available "
            f"(profile={args.profile}, labels={args.label_mode})."
        )

    if manifest_path is not None:
        resolved_manifest_path = (
            (project_root / manifest_path).resolve() if not Path(manifest_path).is_absolute() else Path(manifest_path)
        )
        resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generator": "scripts/syscall_abi/generate_synthetic_inbox.py",
            "config": {
                "inbox_root": str(inbox_root.relative_to(project_root)),
                "count_per_abi": int(args.count_per_abi),
                "min_tokens": int(args.min_tokens),
                "max_tokens": int(args.max_tokens),
                "seed": int(args.seed),
                "prefix": str(args.prefix),
                "profile": str(args.profile),
                "compat_abi": args.compat_abi,
                "compat_max_number": args.compat_max_number,
                "abis": sorted(selected_folders.keys()),
                "label_mode": str(args.label_mode),
                "malware_ratio": float(args.malware_ratio),
                "malware_target_presets": list(selected_malware_target_presets),
            },
            "summary": {
                "generated_total": generated_total,
                "per_abi": per_abi_summary,
            },
            "entries": manifest_entries,
        }
        resolved_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Manifest written to: {resolved_manifest_path}")

    print(f"Generated total files: {generated_total}")


if __name__ == "__main__":
    main()
