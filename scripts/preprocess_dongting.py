import os
import glob
import random
from pathlib import Path

# Mapping of x86_64 syscall names to their official integers
x86_64_syscalls = {
    'read': 0, 'write': 1, 'open': 2, 'close': 3, 'stat': 4, 'fstat': 5, 'lstat': 6, 'poll': 7,
    'lseek': 8, 'mmap': 9, 'mprotect': 10, 'munmap': 11, 'brk': 12, 'rt_sigaction': 13,
    'rt_sigprocmask': 14, 'rt_sigreturn': 15, 'ioctl': 16, 'pread64': 17, 'pwrite64': 18,
    'readv': 19, 'writev': 20, 'access': 21, 'pipe': 22, 'select': 23, 'sched_yield': 24,
    'mremap': 25, 'msync': 26, 'mincore': 27, 'madvise': 28, 'shmget': 29, 'shmat': 30,
    'shmctl': 31, 'dup': 32, 'dup2': 33, 'pause': 34, 'nanosleep': 35, 'getitimer': 36,
    'alarm': 37, 'setitimer': 38, 'getpid': 39, 'sendfile': 40, 'socket': 41, 'connect': 42,
    'accept': 43, 'sendto': 44, 'recvfrom': 45, 'sendmsg': 46, 'recvmsg': 47, 'shutdown': 48,
    'bind': 49, 'listen': 50, 'getsockname': 51, 'getpeername': 52, 'socketpair': 53,
    'setsockopt': 54, 'getsockopt': 55, 'clone': 56, 'fork': 57, 'vfork': 58, 'execve': 59,
    'exit': 60, 'wait4': 61, 'kill': 62, 'uname': 63, 'semget': 64, 'semop': 65, 'semctl': 66,
    'shmdt': 67, 'msgget': 68, 'msgsnd': 69, 'msgrcv': 70, 'msgctl': 71, 'fcntl': 72,
    'flock': 73, 'fsync': 74, 'fdatasync': 75, 'truncate': 76, 'ftruncate': 77, 'getdents': 78,
    'getcwd': 79, 'chdir': 80, 'fchdir': 81, 'rename': 82, 'mkdir': 83, 'rmdir': 84, 'creat': 85,
    'link': 86, 'unlink': 87, 'symlink': 88, 'readlink': 89, 'chmod': 90, 'fchmod': 91,
    'chown': 92, 'fchown': 93, 'lchown': 94, 'umask': 95, 'gettimeofday': 96, 'getrlimit': 97,
    'getrusage': 98, 'sysinfo': 99, 'times': 100, 'ptrace': 101, 'getuid': 102, 'syslog': 103,
    'getgid': 104, 'setuid': 105, 'setgid': 106, 'geteuid': 107, 'getegid': 108, 'setpgid': 109,
    'getppid': 110, 'getpgrp': 111, 'setsid': 112, 'setreuid': 113, 'setregid': 114,
    'getgroups': 115, 'setgroups': 116, 'setresuid': 117, 'setresgid': 118, 'getpgid': 119,
    'setfsuid': 120, 'setfsgid': 121, 'getsid': 122, 'capget': 123, 'capset': 124,
    'rt_sigpending': 125, 'rt_sigtimedwait': 126, 'rt_sigqueueinfo': 127, 'rt_sigsuspend': 128,
    'sigaltstack': 129, 'utime': 130, 'mknod': 131, 'uselib': 132, 'personality': 133,
    'ustat': 134, 'statfs': 135, 'fstatfs': 136, 'sysfs': 137, 'getpriority': 138,
    'setpriority': 139, 'sched_setparam': 140, 'sched_getparam': 141, 'sched_setscheduler': 142,
    'sched_getscheduler': 143, 'sched_get_priority_max': 144, 'sched_get_priority_min': 145,
    'sched_rr_get_interval': 146, 'mlock': 147, 'munlock': 148, 'mlockall': 149, 'munlockall': 150,
    'vhangup': 151, 'modify_ldt': 152, 'pivot_root': 153, '_sysctl': 154, 'prctl': 157,
    'arch_prctl': 158, 'adjtimex': 159, 'setrlimit': 160, 'chroot': 161, 'sync': 162,
    'acct': 163, 'settimeofday': 164, 'mount': 165, 'umount2': 166, 'swapon': 167,
    'swapoff': 168, 'reboot': 169, 'sethostname': 170, 'setdomainname': 171, 'iopl': 172,
    'ioperm': 173, 'create_module': 174, 'init_module': 175, 'delete_module': 176,
    'get_kernel_syms': 177, 'query_module': 178, 'quotactl': 179, 'nfsservctl': 180,
    'getpmsg': 181, 'putpmsg': 182, 'afs_syscall': 183, 'tuxcall': 184, 'security': 185,
    'gettid': 186, 'readahead': 187, 'setxattr': 188, 'lsetxattr': 189, 'fsetxattr': 190,
    'getxattr': 191, 'lgetxattr': 192, 'fgetxattr': 193, 'listxattr': 194, 'llistxattr': 195,
    'flistxattr': 196, 'removexattr': 197, 'lremovexattr': 198, 'fremovexattr': 199,
    'tkill': 200, 'time': 201, 'futex': 202, 'sched_setaffinity': 203, 'sched_getaffinity': 204,
    'set_thread_area': 205, 'io_setup': 206, 'io_destroy': 207, 'io_getevents': 208,
    'io_submit': 209, 'io_cancel': 210, 'get_thread_area': 211, 'lookup_dcookie': 212,
    'epoll_create': 213, 'epoll_ctl_old': 214, 'epoll_wait_old': 215, 'remap_file_pages': 216,
    'getdents64': 217, 'set_tid_address': 218, 'restart_syscall': 219, 'semtimedop': 220,
    'fadvise64': 221, 'timer_create': 222, 'timer_settime': 223, 'timer_gettime': 224,
    'timer_getoverrun': 225, 'timer_delete': 226, 'clock_settime': 227, 'clock_gettime': 228,
    'clock_getres': 229, 'clock_nanosleep': 230, 'exit_group': 231, 'epoll_wait': 232,
    'epoll_ctl': 233, 'tgkill': 234, 'utimes': 235, 'vserver': 236, 'mbind': 237,
    'set_mempolicy': 238, 'get_mempolicy': 239, 'mq_open': 240, 'mq_unlink': 241,
    'mq_timedsend': 242, 'mq_timedreceive': 243, 'mq_notify': 244, 'mq_getsetattr': 245,
    'kexec_load': 246, 'waitid': 247, 'add_key': 248, 'request_key': 249, 'keyctl': 250,
    'ioprio_set': 251, 'ioprio_get': 252, 'inotify_init': 253, 'inotify_add_watch': 254,
    'inotify_rm_watch': 255, 'migrate_pages': 256, 'openat': 257, 'mkdirat': 258,
    'mknodat': 259, 'fchownat': 260, 'futimesat': 261, 'newfstatat': 262, 'unlinkat': 263,
    'renameat': 264, 'linkat': 265, 'symlinkat': 266, 'readlinkat': 267, 'fchmodat': 268,
    'faccessat': 269, 'pselect6': 270, 'ppoll': 271, 'unshare': 272, 'set_robust_list': 273,
    'get_robust_list': 274, 'splice': 275, 'tee': 276, 'sync_file_range': 277,
    'vmsplice': 278, 'move_pages': 279, 'utimensat': 280, 'epoll_pwait': 281,
    'signalfd': 282, 'timerfd_create': 283, 'eventfd': 284, 'fallocate': 285,
    'timerfd_settime': 286, 'timerfd_gettime': 287, 'accept4': 288, 'signalfd4': 289,
    'eventfd2': 290, 'epoll_create1': 291, 'dup3': 292, 'pipe2': 293, 'inotify_init1': 294,
    'preadv': 295, 'pwritev': 296, 'rt_tgsigqueueinfo': 297, 'perf_event_open': 298,
    'recvmmsg': 299, 'sendmmsg': 300, 'fanotify_init': 301, 'fanotify_mark': 302,
    'prlimit64': 303, 'name_to_handle_at': 304, 'open_by_handle_at': 305, 'clock_adjtime': 306,
    'syncfs': 307, 'setns': 308, 'getcpu': 309, 'process_vm_readv': 310,
    'process_vm_writev': 311, 'kcmp': 312, 'finit_module': 313, 'sched_setattr': 314,
    'sched_getattr': 315, 'renameat2': 316, 'seccomp': 317, 'getrandom': 318,
    'memfd_create': 319, 'kexec_file_load': 320, 'bpf': 321, 'execveat': 322,
    'userfaultfd': 323, 'membarrier': 324, 'mlock2': 325, 'copy_file_range': 326,
    'preadv2': 327, 'pwritev2': 328, 'pkey_mprotect': 329, 'pkey_alloc': 330,
    'pkey_free': 331, 'statx': 332, 'io_pgetevents': 333, 'rseq': 334,
    'pidfd_send_signal': 424, 'io_uring_setup': 425, 'io_uring_enter': 426,
    'io_uring_register': 427, 'open_tree': 428, 'move_mount': 429, 'fsopen': 430,
    'fsconfig': 431, 'fsmount': 432, 'fspick': 433, 'pidfd_open': 434, 'clone3': 435,
    'close_range': 436, 'openat2': 437, 'pidfd_getfd': 438, 'faccessat2': 439,
    'process_madvise': 440, 'epoll_pwait2': 441, 'mount_setattr': 442, 'quotactl_fd': 443,
    'landlock_create_ruleset': 444, 'landlock_add_rule': 445, 'landlock_restrict_self': 446,
    'memfd_secret': 447, 'process_mrelease': 448, 'futex_waitv': 449, 'set_mempolicy_home_node': 450,
}

unknown_syscalls = {}
unknown_id_counter = 500  # Start assigning unknown syscalls safely above typical Linux bounds

def get_syscall_id(name):
    global unknown_id_counter
    name = name.strip()
    if not name:
        return None
    if name in x86_64_syscalls:
        return x86_64_syscalls[name]
    if name not in unknown_syscalls:
        unknown_syscalls[name] = unknown_id_counter
        print(f"WARNING Unknown syscall '{name}' encountered. Mapping to ID {unknown_id_counter}.")
        unknown_id_counter += 1
    return unknown_syscalls[name]

def parse_file(file_path):
    syscalls = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # DongTing files typically separate syscalls with '|'
                calls = line.strip().split('|')
                for c in calls:
                    sid = get_syscall_id(c)
                    if sid is not None:
                        syscalls.append(str(sid))
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return syscalls

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(input_dir, '**', '*.log'), recursive=True)

    saved_count = 0
    for i, file_path in enumerate(all_files):
        syscalls = parse_file(file_path)
        if len(syscalls) > 0:
            # Reconstruct filename or just use an index
            base_name = os.path.basename(file_path)
            out_file = os.path.join(output_dir, f"{i}_{base_name}.txt")
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(" ".join(syscalls))
            saved_count += 1

    print(f"INFO Extracted {saved_count} traces from {len(all_files)} files in {input_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42, help="Seed for split reproducibility")
    parser.add_argument("--raw-dir", type=str, default=None, help="DongTing raw dataset directory.")
    parser.add_argument("--output-base", type=str, default=None, help="Output directory for processed split folders.")
    args = parser.parse_args()
    random.seed(args.random_seed)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = Path(project_root) / "datasets"
    raw_dongting_dir = args.raw_dir or str(dataset_root / "64bit" / "DongTing")

    normal_dir = os.path.join(raw_dongting_dir, "Normal_data")
    abnormal_dir = os.path.join(raw_dongting_dir, "Abnormal_data")

    if not os.path.exists(normal_dir) or not os.path.exists(abnormal_dir):
        print(f"Error: {raw_dongting_dir} structure must contain Normal_data and Abnormal_data directories.")
        return

    out_base = args.output_base or str(dataset_root / "64bit")

    train_out = os.path.join(out_base, "Training_Data_Master")
    val_out = os.path.join(out_base, "Validation_Data_Master")
    atk_out = os.path.join(out_base, "Attack_Data_Master")

    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)
    os.makedirs(atk_out, exist_ok=True)

    print("=" * 50)
    print(" DongTing 64-bit Payload Preprocessor ")
    print("=" * 50)


    # Split each normal-domain folder locally so train and validation keep the same domain mix.

    print("INFO Processing stratified 20% train / 80% validation split")

    train_files = []
    val_files = []

    # Walk normal-domain folders.
    for domain_folder in os.listdir(normal_dir):
        folder_path = os.path.join(normal_dir, domain_folder)
        if not os.path.isdir(folder_path):
            continue

        files_in_domain = glob.glob(os.path.join(folder_path, '**', '*.log'), recursive=True)
        files_in_domain.sort() # Ensure reproducible starting state before shuffle
        random.shuffle(files_in_domain)

        split_idx = int(len(files_in_domain) * 0.2) # 20% train bounds
        train_files.extend(files_in_domain[:split_idx])
        val_files.extend(files_in_domain[split_idx:])

        print(f" to Stratified [{domain_folder}]: {split_idx} Train | {len(files_in_domain)-split_idx} Val")

    print(f" to Total Train count: {len(train_files)}")
    print(f" to Total Val count: {len(val_files)}")

    def distribute_files(file_list, target_dir):
        count = 0
        for f in file_list:
            syscalls = parse_file(f)
            if len(syscalls) > 0:
                base_name = os.path.basename(f)
                with open(os.path.join(target_dir, f"{count}_{base_name}.txt"), 'w') as outf:
                    outf.write(" ".join(syscalls))
                count += 1
        return count

    t_c = distribute_files(train_files, train_out)
    v_c = distribute_files(val_files, val_out)
    print(f" INFO Saved {t_c} to {train_out}")
    print(f" INFO Saved {v_c} to {val_out}")

    # Process abnormal data.
    print("\nINFO Processing abnormal data")
    abnormal_files = glob.glob(os.path.join(abnormal_dir, '**', '*.log'), recursive=True)
    a_c = distribute_files(abnormal_files, atk_out)
    print(f" INFO Saved {a_c} out of {len(abnormal_files)} to {atk_out}")

    print("\n==============================================")
    print(" PROCESSING COMPLETE ")
    if unknown_syscalls:
        print(f" Note: Found {len(unknown_syscalls)} unknown strings not in standard x86_64 mapping:")
        for k, v in unknown_syscalls.items():
            print(f" - {k} to {v}")
    else:
        print(" All strings mapped perfectly to standard Linux integers.")
    print("==============================================")

if __name__ == "__main__":
    main()
