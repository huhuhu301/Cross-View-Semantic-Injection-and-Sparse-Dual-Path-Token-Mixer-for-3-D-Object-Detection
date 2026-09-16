#!/usr/bin/env python3
# Modified by RV-SDTM contributors for this public release; see NOTICE.
"""Fail closed on common source-release hygiene mistakes.

The default scope is exactly the files tracked by Git.  ``--include-untracked``
is useful while assembling a fresh release before its initial ``git add``.
Only the Python standard library is required.
"""

import argparse
import hashlib
import ipaddress
import os
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_MAX_FILE_MIB = 5
READ_SAMPLE_BYTES = 8192

# The only binary files in the release are two author-provided qualitative
# result figures. Pinning their bytes keeps the binary-artifact rule intact.
APPROVED_FIGURES = {
    'docs/figures/waymo_long_range.png': (
        '355acab4cc0885e7a8bcd01f15fda6cc6d836229d2144179f7024b2b39f41aba',
        b'\x89PNG\r\n\x1a\n'),
    'docs/figures/waymo_occlusion.png': (
        'ad662463d24effc431a504e9eac32b8c534df5fccb5f7660e3d99ca24d058409',
        b'\x89PNG\r\n\x1a\n'),
}

FORBIDDEN_SUFFIXES = {
    '.7z', '.arrow', '.bin', '.ckpt', '.core', '.dill', '.dll', '.dylib',
    '.db', '.engine', '.feather', '.gz', '.h5', '.hdf5', '.joblib', '.key',
    '.las', '.laz', '.lz4', '.mmap', '.npy', '.npz', '.o', '.obj', '.onnx',
    '.p12', '.parquet', '.pdf', '.pem', '.pfx', '.pickle', '.pkl', '.pt',
    '.pth', '.rar', '.safetensors', '.so', '.sqlite', '.tar', '.tfrecord',
    '.xz', '.zip', '.zst',
}
FORBIDDEN_NAMES = {'core'}
FORBIDDEN_PATH_COMPONENTS = {
    '__pycache__', 'analysis_outputs', 'build', 'checkpoint', 'checkpoints',
    'ckpt', 'data', 'debug_outputs', 'dist', 'explore_outputs', 'log', 'logs',
    'output', 'outputs', 'repro_outputs', 'train_outputs',
}

# Construct these prefixes instead of embedding machine paths verbatim, so the
# scanner does not flag its own source code.
_SLASH = chr(47)
PRIVATE_PATH_PREFIXES = tuple(
    _SLASH + value for value in (
        'root', 'ai35huhao', 'workspace', 'scratch',
    )
)
PRIVATE_PATH_PATTERNS = (
    re.compile(
        r'(?<![A-Za-z0-9_.-])(?:{})'
        r'(?:/[A-Za-z0-9_.@%+,:=~-]+)*'.format(
            '|'.join(re.escape(value) for value in PRIVATE_PATH_PREFIXES))),
    re.compile(
        r'(?<![A-Za-z0-9_.-])/(?:home|Users|mnt)/[A-Za-z0-9_.-]+'
        r'(?:/[A-Za-z0-9_.@%+,:=~-]+)*'),
    re.compile(r'(?<![%A-Za-z0-9_.-])[A-Za-z]:[\\/][^\s\'"`<>|]+'),
)
IPV4_PATTERN = re.compile(
    r'(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])')
SECRET_PATTERNS = (
    ('private key', re.compile(
        r'-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----')),
    ('AWS access key', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
    ('GitHub token', re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,255}\b')),
    ('OpenAI-style token', re.compile(r'\bsk-[A-Za-z0-9_-]{20,255}\b')),
    ('assigned credential', re.compile(
        r'(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|'
        r'password)\b\s*[:=]\s*[\'\"][^\'\"]{8,}[\'\"]')),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Scan Git-tracked release sources for unsafe artifacts')
    parser.add_argument(
        '--root', type=Path,
        default=Path(__file__).resolve().parents[1],
        help='release repository root (default: parent of tools/)')
    parser.add_argument(
        '--max-file-mib', type=int, default=DEFAULT_MAX_FILE_MIB,
        help='reject individual files larger than this many MiB')
    parser.add_argument(
        '--include-untracked', action='store_true',
        help='also scan untracked, non-ignored files')
    return parser.parse_args()


def git_files(root, include_untracked):
    command = ['git', '-C', str(root), 'ls-files', '-z', '--cached']
    if include_untracked:
        command.extend(['--others', '--exclude-standard'])
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        detail = result.stderr.decode('utf-8', errors='replace').strip()
        raise RuntimeError('git ls-files failed: {}'.format(detail))
    return sorted(
        Path(value.decode('utf-8', errors='surrogateescape'))
        for value in result.stdout.split(b'\0') if value)


def is_rfc1918(value):
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if address.version != 4:
        return False
    first, second = (int(part) for part in value.split('.')[:2])
    return (
        first == 10
        or (first == 172 and 16 <= second <= 31)
        or (first == 192 and second == 168)
    )


def scan_text(relative_path, text):
    findings = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for pattern in PRIVATE_PATH_PATTERNS:
            match = pattern.search(line)
            if match:
                findings.append(
                    '{}:{}: private absolute path {}'.format(
                        relative_path, line_number, repr(match.group(0))))
        for match in IPV4_PATTERN.finditer(line):
            if is_rfc1918(match.group(0)):
                findings.append(
                    '{}:{}: RFC1918 address {}'.format(
                        relative_path, line_number, match.group(0)))
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(
                    '{}:{}: possible {}'.format(
                        relative_path, line_number, label))
    return findings


def scan_file(root, relative_path, max_bytes):
    findings = []
    path = root / relative_path
    forbidden_components = sorted(
        set(relative_path.parts) & FORBIDDEN_PATH_COMPONENTS)
    if forbidden_components:
        findings.append(
            '{}: forbidden generated-artifact path component(s): {}'.format(
                relative_path, forbidden_components))
    if not path.exists() and not path.is_symlink():
        return ['{}: tracked path is missing'.format(relative_path)]
    if path.is_symlink():
        target = os.readlink(str(path))
        resolved_target = (path.parent / target).resolve()
        try:
            resolved_target.relative_to(root.resolve())
        except ValueError:
            findings.append(
                '{}: symlink escapes repository: {}'.format(
                    relative_path, target))
        return findings
    if not path.is_file():
        return ['{}: tracked path is not a regular file'.format(relative_path)]

    size = path.stat().st_size
    if size > max_bytes:
        findings.append(
            '{}: file is {:.2f} MiB (limit {:.2f} MiB)'.format(
                relative_path, size / (1024 ** 2), max_bytes / (1024 ** 2)))
        return findings

    suffix = path.suffix.lower()
    if suffix in FORBIDDEN_SUFFIXES or path.name.lower() in FORBIDDEN_NAMES:
        findings.append(
            '{}: forbidden generated/binary artifact'.format(relative_path))

    data = path.read_bytes()
    approved_figure = APPROVED_FIGURES.get(relative_path.as_posix())
    if approved_figure is not None:
        expected_hash, signature = approved_figure
        if not data.startswith(signature):
            findings.append('{}: invalid approved figure format'.format(
                relative_path))
        if hashlib.sha256(data).hexdigest() != expected_hash:
            findings.append('{}: approved figure hash mismatch'.format(
                relative_path))
        return findings
    if b'\0' in data[:READ_SAMPLE_BYTES]:
        findings.append('{}: binary content detected'.format(relative_path))
        return findings
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        findings.append('{}: source file is not UTF-8 text'.format(relative_path))
        return findings
    findings.extend(scan_text(relative_path, text))
    return findings


def main():
    args = parse_args()
    root = args.root.resolve()
    if args.max_file_mib <= 0:
        print('[FAIL] --max-file-mib must be positive', file=sys.stderr)
        return 2
    try:
        paths = git_files(root, args.include_untracked)
    except RuntimeError as error:
        print('[FAIL] {}'.format(error), file=sys.stderr)
        return 2
    if not paths:
        print(
            '[FAIL] no files selected; add files to Git or pass '
            '--include-untracked', file=sys.stderr)
        return 1

    max_bytes = args.max_file_mib * 1024 * 1024
    findings = []
    total_bytes = 0
    for relative_path in paths:
        path = root / relative_path
        if path.is_file() and not path.is_symlink():
            total_bytes += path.stat().st_size
        findings.extend(scan_file(root, relative_path, max_bytes))

    if findings:
        for finding in findings:
            print('[FAIL] {}'.format(finding))
        print('[SUMMARY] {} files scanned, {} findings'.format(
            len(paths), len(findings)))
        return 1
    print('[PASS] {} Git source files scanned ({:.2f} MiB); no release '
          'hygiene findings'.format(len(paths), total_bytes / (1024 ** 2)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
