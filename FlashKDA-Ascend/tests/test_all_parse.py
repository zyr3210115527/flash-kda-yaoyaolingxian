"""Every test and benchmark in this repo parses.

Four diagnostic probes -- check_prepare, dump_neumann, per_chunk_err and
dump_k2_chunk0 -- sat broken for days with an IndentationError, because an
automated edit put a module-level import inside a function body. They are listed
in STATUS.md as tests the whole time. Nobody noticed, because the shape suite
was green and these are only reached for when something is *already* wrong --
so the tools for diagnosing a failure were themselves broken, and would have
been discovered at the worst possible moment.

This needs no hardware and takes under a second. Run it before trusting that a
probe "found nothing".
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    bad = []
    checked = 0
    for sub in ('tests', 'benchmarks', 'flash_kda'):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith('.py'):
                continue
            path = os.path.join(d, name)
            checked += 1
            # compile() rather than py_compile: it needs no output file, so
            # there is no .pyc to write and nothing to clean up.
            try:
                with open(path, 'rb') as fh:
                    compile(fh.read(), path, 'exec')
            except SyntaxError as e:
                bad.append((os.path.join(sub, name), f"line {e.lineno}: {e.msg}"))

    for path, err in bad:
        print(f"  BROKEN  {path}\n            {err}")
    print(f"{checked - len(bad)}/{checked} files parse")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
