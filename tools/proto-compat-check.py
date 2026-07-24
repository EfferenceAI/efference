#!/usr/bin/env python3
# proto-compat-check.py OLD.proto NEW.proto
#
# Wire-compatibility gate for the control protocol. Enforces the ef.proto rule
# (line 9: "ADDITIVE ONLY, new field numbers only, never renumber/reuse,
# `reserved` for removals"). Compares a previously released proto against the
# one being tagged and FAILS (exit 1) if any existing field/verb NUMBER is:
#
#   * subtracted  - the number is gone and not marked `reserved`            (ERROR)
#   * renumbered  - shows up as the old number vanishing                    (ERROR)
#   * overwritten - the number now carries a DIFFERENT TYPE (reused)        (ERROR)
#   * scope gone  - the whole message/enum was deleted                      (ERROR)
#
# A field renamed in place (same number, same type, new identifier) is a
# source/JSON break but not a wire-tag subtraction or overwrite, so it is
# reported as a WARNING and does not fail the gate. Enum value renames land
# here too (an enum number's meaning is its ordinal, which is unchanged).
#
# Field numbers are scoped per message; the Request/Response `oneof body` is
# transparent (its verb numbers belong to the enclosing message); nested enums
# and messages get their own number space (e.g. StreamTarget.Kind).
import re, sys

SPECIAL = set('{};=,')

def tokenize(text):
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)   # block comments
    text = re.sub(r'//[^\n]*', ' ', text)                # line comments
    toks, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1; continue
        if c == '"':                                     # string literal -> one token
            j = i + 1
            while j < n and text[j] != '"':
                j += 1
            toks.append(text[i:j+1]); i = j + 1; continue
        if c in SPECIAL:
            toks.append(c); i += 1; continue
        j = i                                            # identifier / number / dotted.type
        while j < n and not text[j].isspace() and text[j] not in SPECIAL and text[j] != '"':
            j += 1
        toks.append(text[i:j]); i = j
    return toks

def parse(text):
    """Return {fqname: {'kind', 'numbers': {num: (name, type)}, 'reserved': set()}}."""
    scopes, stack, real_fq, buf = {}, [], [], []

    def cur_real():
        for fr in reversed(stack):
            if fr['real']:
                return fr
        return None

    def flush():
        nonlocal buf
        if not buf:
            return
        fr = cur_real()
        if buf[0] == 'reserved':
            if fr is not None:
                for k, t in enumerate(buf):                       # ranges: "a to b"
                    if t == 'to' and 0 < k < len(buf) - 1 \
                       and re.fullmatch(r'\d+', buf[k-1]) and re.fullmatch(r'\d+', buf[k+1]):
                        for v in range(int(buf[k-1]), int(buf[k+1]) + 1):
                            fr['scope']['reserved'].add(v)
                for t in buf[1:]:
                    if re.fullmatch(r'\d+', t):
                        fr['scope']['reserved'].add(int(t))
            buf = []; return
        if '=' in buf and fr is not None:
            ei = buf.index('=')
            if ei >= 1 and ei + 1 < len(buf) and re.fullmatch(r'\d+', buf[ei+1]):
                num, name = int(buf[ei+1]), buf[ei-1]
                typ = 'enum' if fr['kind'] == 'enum' else ' '.join(buf[:ei-1])
                fr['scope']['numbers'][num] = (name, typ)
        buf = []

    for t in tokenize(text):
        if t == '{':
            kind, name, real = 'other', None, False
            if buf and buf[0] in ('message', 'enum'):
                kind, real = buf[0], True
                name = buf[1] if len(buf) > 1 else '?'
            elif buf and buf[0] == 'oneof':
                kind = 'oneof'
            scope = None
            if real:
                parent = real_fq[-1] if real_fq else None
                fq = (parent + '.' + name) if parent else name
                scope = scopes.setdefault(fq, {'kind': kind, 'numbers': {}, 'reserved': set()})
                real_fq.append(fq)
            stack.append({'kind': kind, 'name': name, 'real': real, 'scope': scope})
            buf = []
        elif t == '}':
            if stack:
                fr = stack.pop()
                if fr['real'] and real_fq:
                    real_fq.pop()
            buf = []
        elif t == ';':
            flush()
        else:
            buf.append(t)
    return scopes

def compare(old, new):
    errors, warnings = [], []
    for scope, od in old.items():
        nd = new.get(scope)
        if nd is None:
            fields = ', '.join(f'{n}={nm}' for n, (nm, _) in sorted(od['numbers'].items())) or '(empty)'
            errors.append(f'SCOPE REMOVED : "{scope}" deleted (tags: {fields})')
            continue
        for num, (oname, otype) in sorted(od['numbers'].items()):
            if num not in nd['numbers']:
                if num in nd['reserved']:
                    continue                                       # properly retired
                errors.append(f'TAG REMOVED   : {scope} #{num} ({oname}: {otype}) '
                              f'gone and not marked `reserved`')
                continue
            nname, ntype = nd['numbers'][num]
            if ntype != otype:
                errors.append(f'TAG OVERWRITTEN: {scope} #{num} was ({oname}: {otype}) '
                              f'now ({nname}: {ntype})')
            elif nname != oname:
                warnings.append(f'renamed       : {scope} #{num} ({otype}) {oname} -> {nname} '
                                f'(same tag+type; wire-safe, source/JSON break)')
    return errors, warnings

def main():
    if len(sys.argv) != 3:
        sys.exit('usage: proto-compat-check.py OLD.proto NEW.proto')
    old = parse(open(sys.argv[1]).read())
    new = parse(open(sys.argv[2]).read())
    errors, warnings = compare(old, new)
    for w in warnings:
        print('WARN  ' + w)
    for e in errors:
        print('ERROR ' + e)
    if errors:
        print(f'\n!! proto-compat: FAILED - {len(errors)} wire-breaking change(s); '
              f'publish aborted. Retire a tag with `reserved <n>;`, never remove or reuse it.')
        sys.exit(1)
    print(f'>> proto-compat: OK - no removed/renumbered/reused tags '
          f'({len(warnings)} rename warning(s))')

if __name__ == '__main__':
    main()
