"""CWE-style paired (vulnerable, fixed) C function templates used to synthesize a labeled
training corpus when no real manually-labeled dataset is available (see data/download.py).

Each generator returns (vulnerable_source, safe_source, base_name). Randomized identifier names,
sizes and small structural variations (extra benign locals/statements) are injected via `rng` so
that many distinct, non-duplicate functions can be sampled from the same template.
"""
from __future__ import annotations

import random

_TYPES = ["int", "long", "short", "unsigned int", "size_t"]
_VAR_POOL = ["a", "b", "c", "x", "y", "z", "n", "len", "idx", "tmp", "val", "ret", "count", "buf_len"]


def _names(rng: random.Random, k: int) -> list[str]:
    pool = rng.sample(_VAR_POOL, k=min(k, len(_VAR_POOL)))
    while len(pool) < k:
        pool.append(f"v{rng.randint(0, 9999)}")
    return pool


def _filler(rng: random.Random, var: str) -> str:
    """A benign extra statement to perturb node count / token sequence across samples."""
    n = rng.randint(0, 1000)
    choices = [
        f"int {var}_pad = {n};",
        f"int {var}_pad2 = {n} * 2;",
        f"int {var}_pad3 = {n} + 7;",
    ]
    return rng.choice(choices)


def gen_integer_overflow(rng: random.Random):
    t = rng.choice(["short", "int"])
    a, b = _names(rng, 2)
    fname = f"add_{a}_{b}_{rng.randint(0, 99999)}"
    vuln = f"""
{t} {fname}({t} {b}) {{
    {t} {a} = 32767;
    if ({b} > 0) {{
        {a} = {a} + {b};
    }}
    return {a};
}}
"""
    safe = f"""
{t} {fname}({t} {b}) {{
    long {a} = 32767;
    if ({b} > 0) {{
        {a} = {a} + (long){b};
        if ({a} > 32767) {{
            {a} = 32767;
        }}
    }}
    return ({t}){a};
}}
"""
    return vuln, safe, fname


def gen_buffer_overflow(rng: random.Random):
    n = rng.randint(8, 64)
    src, dst, len_v = _names(rng, 3)
    fname = f"copy_buf_{rng.randint(0, 99999)}"
    vuln = f"""
void {fname}(const char *{src}) {{
    char {dst}[{n}];
    strcpy({dst}, {src});
    {_filler(rng, dst)}
}}
"""
    safe = f"""
void {fname}(const char *{src}) {{
    char {dst}[{n}];
    size_t {len_v} = strlen({src});
    if ({len_v} >= sizeof({dst})) {{
        {len_v} = sizeof({dst}) - 1;
    }}
    strncpy({dst}, {src}, {len_v});
    {dst}[{len_v}] = '\\0';
    {_filler(rng, dst)}
}}
"""
    return vuln, safe, fname


def gen_null_deref(rng: random.Random):
    p, n = _names(rng, 2)
    fname = f"alloc_use_{rng.randint(0, 99999)}"
    vuln = f"""
int {fname}(int {n}) {{
    int *{p} = (int *)malloc({n} * sizeof(int));
    {p}[0] = {n};
    {_filler(rng, p)}
    return {p}[0];
}}
"""
    safe = f"""
int {fname}(int {n}) {{
    int *{p} = (int *)malloc({n} * sizeof(int));
    if ({p} == NULL) {{
        return -1;
    }}
    {p}[0] = {n};
    {_filler(rng, p)}
    return {p}[0];
}}
"""
    return vuln, safe, fname


def gen_use_after_free(rng: random.Random):
    p, n = _names(rng, 2)
    fname = f"free_use_{rng.randint(0, 99999)}"
    vuln = f"""
void {fname}(int {n}) {{
    int *{p} = (int *)malloc({n} * sizeof(int));
    free({p});
    {p}[0] = {n};
    {_filler(rng, p)}
}}
"""
    safe = f"""
void {fname}(int {n}) {{
    int *{p} = (int *)malloc({n} * sizeof(int));
    free({p});
    {p} = NULL;
    {_filler(rng, p)}
}}
"""
    return vuln, safe, fname


def gen_format_string(rng: random.Random):
    s = _names(rng, 1)[0]
    fname = f"log_msg_{rng.randint(0, 99999)}"
    vuln = f"""
void {fname}(const char *{s}) {{
    printf({s});
    {_filler(rng, s)}
}}
"""
    safe = f"""
void {fname}(const char *{s}) {{
    printf("%s", {s});
    {_filler(rng, s)}
}}
"""
    return vuln, safe, fname


def gen_off_by_one(rng: random.Random):
    n = rng.randint(8, 64)
    arr, i, val = _names(rng, 3)
    fname = f"fill_array_{rng.randint(0, 99999)}"
    vuln = f"""
void {fname}(int {val}) {{
    int {arr}[{n}];
    int {i};
    for ({i} = 0; {i} <= {n}; {i}++) {{
        {arr}[{i}] = {val};
    }}
}}
"""
    safe = f"""
void {fname}(int {val}) {{
    int {arr}[{n}];
    int {i};
    for ({i} = 0; {i} < {n}; {i}++) {{
        {arr}[{i}] = {val};
    }}
}}
"""
    return vuln, safe, fname


def gen_division_by_zero(rng: random.Random):
    a, b = _names(rng, 2)
    fname = f"divide_{rng.randint(0, 99999)}"
    vuln = f"""
int {fname}(int {a}, int {b}) {{
    int result = {a} / {b};
    {_filler(rng, a)}
    return result;
}}
"""
    safe = f"""
int {fname}(int {a}, int {b}) {{
    if ({b} == 0) {{
        return 0;
    }}
    int result = {a} / {b};
    {_filler(rng, a)}
    return result;
}}
"""
    return vuln, safe, fname


def gen_uninitialized_var(rng: random.Random):
    t = rng.choice(_TYPES)
    v, cond = _names(rng, 2)
    fname = f"compute_{rng.randint(0, 99999)}"
    vuln = f"""
{t} {fname}(int {cond}) {{
    {t} {v};
    if ({cond} > 0) {{
        {v} = {cond} * 2;
    }}
    return {v};
}}
"""
    safe = f"""
{t} {fname}(int {cond}) {{
    {t} {v} = 0;
    if ({cond} > 0) {{
        {v} = {cond} * 2;
    }}
    return {v};
}}
"""
    return vuln, safe, fname


def gen_double_free(rng: random.Random):
    p, n = _names(rng, 2)
    fname = f"cleanup_{rng.randint(0, 99999)}"
    vuln = f"""
void {fname}(int {n}, int flag) {{
    int *{p} = (int *)malloc({n} * sizeof(int));
    free({p});
    if (flag) {{
        free({p});
    }}
}}
"""
    safe = f"""
void {fname}(int {n}, int flag) {{
    int *{p} = (int *)malloc({n} * sizeof(int));
    free({p});
    {p} = NULL;
    if (flag) {{
        if ({p} != NULL) {{
            free({p});
        }}
    }}
}}
"""
    return vuln, safe, fname


def gen_memory_leak(rng: random.Random):
    p, n = _names(rng, 2)
    fname = f"process_{rng.randint(0, 99999)}"
    vuln = f"""
int {fname}(int {n}) {{
    int *{p} = (int *)malloc({n} * sizeof(int));
    if ({n} < 0) {{
        return -1;
    }}
    {p}[0] = {n};
    free({p});
    return 0;
}}
"""
    safe = f"""
int {fname}(int {n}) {{
    int *{p} = (int *)malloc({n} * sizeof(int));
    if ({n} < 0) {{
        free({p});
        return -1;
    }}
    {p}[0] = {n};
    free({p});
    return 0;
}}
"""
    return vuln, safe, fname


TEMPLATES = [
    gen_integer_overflow,
    gen_buffer_overflow,
    gen_null_deref,
    gen_use_after_free,
    gen_format_string,
    gen_off_by_one,
    gen_division_by_zero,
    gen_uninitialized_var,
    gen_double_free,
    gen_memory_leak,
]
