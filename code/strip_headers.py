import ast, pathlib, sys
folder = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "../nevus_shift_code/code")
done = 0
for p in sorted(folder.iterdir()):
    if p.suffix not in (".py", ".sh"):
        continue
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    keep_first = lines[:1] if lines and lines[0].startswith("#!") else []
    body = lines[len(keep_first):]
    if p.suffix == ".py":
        try:
            tree = ast.parse("".join(body))
            first = tree.body[0] if tree.body else None
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str):
                body = body[first.end_lineno:]
        except SyntaxError:
            pass
    i = 0
    while i < len(body) and (body[i].strip() == "" or body[i].lstrip().startswith("#")):
        i += 1
    new = "".join(keep_first + body[i:])
    if p.suffix == ".py":
        compile(new, str(p), "exec")
    p.write_text(new, encoding="utf-8"); done += 1
print(f"headers removed from {done} files in {folder}")
