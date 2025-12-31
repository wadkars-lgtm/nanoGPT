import sys
from ast import literal_eval

def parse_value(val: str):
    try:
        return literal_eval(val)
    except (SyntaxError, ValueError):
        return val

def coerce(value, target_type):
    # best-effort coercion for common cases
    if target_type is bool and isinstance(value, int):
        return bool(value)
    if target_type is float and isinstance(value, int):
        return float(value)
    return target_type(value)

for arg in sys.argv[1:]:
    if '=' not in arg:
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read(), globals())
    else:
        assert arg.startswith('--')
        key, val = arg.split('=', 1)
        key = key[2:]
        attempt = parse_value(val)

        if key in globals():
            # If types differ, try to coerce to existing type instead of asserting
            existing = globals()[key]
            if type(attempt) is not type(existing):
                try:
                    attempt = coerce(attempt, type(existing))
                except Exception:
                    raise TypeError(
                        f"Type mismatch for --{key}={val}. "
                        f"Config has {type(existing).__name__}, you gave {type(attempt).__name__}."
                    )
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            # NEW: allow brand-new keys (e.g., n_kv_head)
            print(f"Setting new config key: {key} = {attempt}")
            globals()[key] = attempt
