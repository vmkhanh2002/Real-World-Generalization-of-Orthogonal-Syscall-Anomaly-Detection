"""Load canonical syscall ABI mappings and expose lookup helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ABI_DISPLAY_NAMES = {
 "i386": "Linux x86_32/i386",
 "x86_64": "Linux x86_64",
 "arm": "Linux ARM EABI",
 "arm64_asm_generic": "Linux arm64/asm-generic",
}

ABI_ALIASES = {
 "i386": "i386",
 "x86": "i386",
 "x86_32": "i386",
 "x86-32": "i386",
 "x8632": "i386",
 "ia32": "i386",
 "linux_i386": "i386",
 "linux_x86_32": "i386",
 "x86_64": "x86_64",
 "x64": "x86_64",
 "amd64": "x86_64",
 "linux_x86_64": "x86_64",
 "arm": "arm",
 "arm32": "arm",
 "arm_eabi": "arm",
 "linux_arm": "arm",
 "arm64": "arm64_asm_generic",
 "aarch64": "arm64_asm_generic",
 "asm_generic": "arm64_asm_generic",
 "asm-generic": "arm64_asm_generic",
 "arm64_asm_generic": "arm64_asm_generic",
 "arm64/asm-generic": "arm64_asm_generic",
 "linux_arm64": "arm64_asm_generic",
}

DEFAULT_DATASET_ABI = {
 "32bit": "i386",
 "64bit": "x86_64",
}

COMMON_USERSPACE_SYSCALLS = {
 "restart_syscall",
 "read",
 "write",
 "open",
 "openat",
 "close",
 "stat",
 "stat64",
 "lstat",
 "lstat64",
 "fstat",
 "fstat64",
 "newfstatat",
 "fstatat64",
 "mmap",
 "mmap2",
 "mprotect",
 "munmap",
 "brk",
 "access",
 "faccessat",
 "ioctl",
 "poll",
 "ppoll",
 "select",
 "newselect",
 "getdents",
 "getdents64",
 "dup",
 "dup2",
 "dup3",
 "clone",
 "fork",
 "vfork",
 "execve",
 "exit",
 "exit_group",
 "waitpid",
 "wait4",
 "rt_sigaction",
 "rt_sigprocmask",
 "sigaction",
 "sigprocmask",
 "sigreturn",
 "rt_sigreturn",
 "futex",
 "clock_gettime",
 "clock_getres",
 "socketcall",
 "socket",
 "connect",
 "accept",
 "accept4",
 "recv",
 "recvfrom",
 "recvmsg",
 "sendto",
 "sendmsg",
 "bind",
 "listen",
 "shutdown",
 "getsockopt",
 "setsockopt",
 "getsockname",
 "getpeername",
 "gettimeofday",
 "sched_getparam",
 "sched_getscheduler",
 "sched_setparam",
 "sched_setscheduler",
 "sched_rr_get_interval",
 "sched_getaffinity",
 "set_tid_address",
 "set_robust_list",
 "get_robust_list",
 "mkdir",
 "rmdir",
 "chmod",
 "unlink",
 "link",
 "rename",
 "kill",
 "pipe",
 "pipe2",
 "eventfd",
 "epoll_wait",
 "epoll_ctl",
 "epoll_create",
 "epoll_create1",
 "pread64",
 "preadv",
 "pwrite64",
 "pwritev",
 "prlimit64",
 "nanosleep",
 "pause",
 "setitimer",
 "getitimer",
 "adjtimex",
}

ARCH_DISCRIMINATIVE_IDS = (
    63,
    102,
    114,
    120,
    168,
    175,
    192,
    195,
    197,
    220,
    221,
    240,
    252,
    265,
    295,
    300,
    309,
    311,
    331,
    340,
)

ABI_MAPPING_FILES = {
 "i386": "i386.json",
 "x86_64": "x86_64.json",
 "arm": "arm_eabi.json",
 "arm64_asm_generic": "arm64_asm_generic.json",
}


@dataclass
class SyscallAbiMapping:
    abi: str
    display_name: str
    source_path: Path
    name_to_numbers: dict[str, tuple[int, ...]]
    number_to_names: dict[int, set[str]]

    def numbers_for_name(self, name: str) -> set[int]:
        return set(self.name_to_numbers.get(str(name).strip(), ()))

    def names_for_number(self, number: int) -> set[str]:
        return set(self.number_to_names.get(int(number), set()))

    def primary_number_for_name(self, name: str) -> int | None:
        numbers = self.name_to_numbers.get(str(name).strip(), ())
        return numbers[0] if numbers else None

    def primary_name_for_number(self, number: int) -> str | None:
        names = sorted(self.names_for_number(number))
        return names[0] if names else None


def normalize_abi_name(value: str | None) -> str:
    if value is None:
        raise ValueError("ABI name cannot be None")
    key = re.sub(r"[^a-z0-9_/-]+", "", value.strip().lower().replace(" ", "_"))
    if key not in ABI_ALIASES:
        raise ValueError(f"Unsupported syscall ABI '{value}'. Expected one of: {sorted(ABI_DISPLAY_NAMES)}")
    return ABI_ALIASES[key]


def get_default_abi_for_dataset(dataset: str) -> str:
    return DEFAULT_DATASET_ABI.get(str(dataset).strip(), "i386")


def is_common_userspace_name(name: str) -> bool:
    return name in COMMON_USERSPACE_SYSCALLS


def get_supported_abis() -> list[str]:
    return list(ABI_DISPLAY_NAMES)


def get_supported_abi_display_names() -> dict[str, str]:
    return dict(ABI_DISPLAY_NAMES)


def resolve_project_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return Path(__file__).resolve().parents[2]


def get_registry_dir(project_root: Path | None = None) -> Path:
    return resolve_project_root(project_root) / "scripts" / "syscall_abi" / "mappings"


def resolve_mapping_json_path(abi: str, project_root: Path | None = None) -> Path:
    normalized = normalize_abi_name(abi)
    registry_dir = get_registry_dir(project_root)
    filename = ABI_MAPPING_FILES[normalized]
    path = registry_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Syscall ABI mapping JSON not found for {normalized}: {path}")
    return path


def _normalize_name_to_numbers(raw_name_to_numbers: dict[str, object]) -> dict[str, tuple[int, ...]]:
    normalized: dict[str, tuple[int, ...]] = {}
    for raw_name, raw_values in raw_name_to_numbers.items():
        name = str(raw_name).strip()
        if not name:
            continue
        if isinstance(raw_values, (list, tuple, set)):
            values = raw_values
        else:
            values = [raw_values]
        numbers: list[int] = []
        seen: set[int] = set()
        for value in values:
            number = int(value)
            if number in seen:
                continue
            seen.add(number)
            numbers.append(number)
        if numbers:
            normalized[name] = tuple(numbers)
    return normalized


def _build_reverse_mapping(name_to_numbers: dict[str, tuple[int, ...]]) -> dict[int, set[str]]:
    reverse: dict[int, set[str]] = {}
    for name, numbers in name_to_numbers.items():
        for number in numbers:
            reverse.setdefault(int(number), set()).add(name)
    return reverse


def load_syscall_abi_mapping(abi: str, project_root: Path | None = None) -> SyscallAbiMapping:
    path = resolve_mapping_json_path(abi, project_root=project_root)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    normalized_abi = normalize_abi_name(payload.get("abi", abi))
    name_to_numbers = _normalize_name_to_numbers(payload["name_to_numbers"])
    number_to_names = _build_reverse_mapping(name_to_numbers)
    return SyscallAbiMapping(
        abi=normalized_abi,
        display_name=payload.get("display_name", ABI_DISPLAY_NAMES[normalized_abi]),
        source_path=path,
        name_to_numbers=name_to_numbers,
        number_to_names=number_to_names,
    )


def load_all_syscall_abi_mappings(project_root: Path | None = None) -> dict[str, SyscallAbiMapping]:
    return {
        abi: load_syscall_abi_mapping(abi, project_root=project_root)
        for abi in ABI_DISPLAY_NAMES
    }


def get_syscall_table(abi: str, project_root: Path | None = None) -> dict[str, int]:
    mapping = load_syscall_abi_mapping(abi, project_root=project_root)
    table: dict[str, int] = {}
    for name in sorted(mapping.name_to_numbers):
        number = mapping.primary_number_for_name(name)
        if number is not None:
            table[name] = number
    return table


def get_syscall_maps(abi: str, project_root: Path | None = None) -> tuple[dict[str, set[int]], dict[int, set[str]]]:
    mapping = load_syscall_abi_mapping(abi, project_root=project_root)
    return (
        {name: set(numbers) for name, numbers in mapping.name_to_numbers.items()},
        {number: set(names) for number, names in mapping.number_to_names.items()},
    )


def lookup_syscall_name(abi: str, name: str, project_root: Path | None = None) -> set[int]:
    mapping = load_syscall_abi_mapping(abi, project_root=project_root)
    return mapping.numbers_for_name(name)


def lookup_syscall_number(abi: str, number: int, project_root: Path | None = None) -> set[str]:
    mapping = load_syscall_abi_mapping(abi, project_root=project_root)
    return mapping.names_for_number(number)
