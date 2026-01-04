# bench/configurator.py
import sys
from ast import literal_eval


_TRUE = {"1", "true", "yes", "y", "t", "on"}
_FALSE = {"0", "false", "no", "n", "f", "off", ""}  # allow empty as false
_NONE = {"none", "null", "nil"}


def parse_bool(val: str) -> bool | None:
    s = str(val).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    if s in _NONE:
        return None
    return None


def parse_value(val: str):
    """
    Best-effort parsing for CLI values.

    Priority:
      1) common bool tokens (true/false/1/0/yes/no/on/off)
      2) None tokens (none/null)
      3) python literal_eval (numbers, lists, dicts, quoted strings, etc.)
      4) raw string fallback
    """
    b = parse_bool(val)
    if b is not None:
        return b

    s = str(val).strip().lower()
    if s in _NONE:
        return None

    try:
        return literal_eval(val)
    except (SyntaxError, ValueError):
        return val


def coerce(value, target_type):
    """
    Coerce CLI-parsed `value` into the type of the existing config var.
    Avoids Python's bool("false") == True bug by explicit handling.
    """
    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(int(value))
        if isinstance(value, str):
            b = parse_bool(value)
            if b is None:
                raise TypeError(f"Cannot coerce string to bool: {value!r}")
            return b
        raise TypeError(f"Cannot coerce {type(value).__name__} to bool")

    if target_type is int:
        if isinstance(value, bool):
            return int(value)
        return int(value)

    if target_type is float:
        if isinstance(value, bool):
            return float(int(value))
        return float(value)

    # For strings, make sure we keep stringy things stringy
    if target_type is str:
        return str(value)

    # Generic fallback (e.g., Path-like, custom types)
    return target_type(value)


for arg in sys.argv[1:]:
    if "=" not in arg:
        # config file
        assert not arg.startswith("--")
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file, encoding="utf-8") as f:
            print(f.read())
        exec(open(config_file, "r", encoding="utf-8").read(), globals())
    else:
        # CLI override --key=value
        assert arg.startswith("--")
        key, val = arg.split("=", 1)
        key = key[2:]
        attempt = parse_value(val)

        if key in globals():
            existing = globals()[key]

            # If the existing value is None, we cannot infer a target type.
            # In that case, accept the parsed attempt as-is.
            if existing is None:
                coerced = attempt
            else:
                # Coerce to the existing type if needed
                if type(attempt) is not type(existing):
                    try:
                        coerced = coerce(attempt, type(existing))
                    except Exception as e:
                        raise TypeError(
                            f"Type mismatch for --{key}={val}. "
                            f"Config has {type(existing).__name__}, you gave {type(attempt).__name__}. "
                            f"Coercion failed: {e}"
                        ) from e
                else:
                    coerced = attempt

            print(f"Overriding: {key} = {coerced}")
            globals()[key] = coerced
        else:
            # allow brand-new keys
            print(f"Setting new config key: {key} = {attempt}")
            globals()[key] = attempt
