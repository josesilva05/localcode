"""Writes the coding tasks of the agent benchmark (tools/bench/bench.py) into tools/bench/tasks/.

Each task is a folder: the project's files, prompt.txt (what the agent is asked, in Portuguese as the
user writes), and check.py (exit code 0 when the task is done: it runs the project's tests and may
look at the code). The agents never see check.py: bench.py copies the rest into a fresh folder.
"""
import os
import textwrap

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")


def task(name, files, prompt, check):
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    for path, text in files.items():
        p = os.path.join(d, "project", path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(textwrap.dedent(text).lstrip("\n"))
    with open(os.path.join(d, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt.strip() + "\n")
    with open(os.path.join(d, "check.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(textwrap.dedent(check).lstrip("\n"))


# 1. two bugs, a test that shows them
task("bugfix_stats", {
    "stats.py": '''
        def mediana(valores):
            """Mediana de uma lista de numeros."""
            v = sorted(valores)
            n = len(v)
            meio = n // 2
            if n % 2 == 0:
                return v[meio]
            return v[meio]


        def variancia(valores):
            """Variancia populacional de uma lista de numeros."""
            m = sum(valores) / len(valores)
            return sum((x - m) ** 2 for x in valores) / (len(valores) - 1)
        ''',
    "test_stats.py": '''
        from stats import mediana, variancia

        assert mediana([3, 1, 2]) == 2
        assert mediana([4, 1, 3, 2]) == 2.5
        assert abs(variancia([1, 2, 3, 4]) - 1.25) < 1e-9
        print("OK")
        ''',
}, "O teste test_stats.py falha. Rode python test_stats.py, ache os erros em stats.py e corrija ate passar.", '''
    import subprocess, sys
    r = subprocess.run([sys.executable, "test_stats.py"], capture_output=True, text=True)
    sys.exit(0 if r.returncode == 0 and "OK" in r.stdout else 1)
    ''')

# 2. a class to finish, from its tests
task("feature_inventory", {
    "inventory.py": '''
        class Inventory:
            """Estoque: nome do produto -> (quantidade, preco unitario)."""

            def __init__(self):
                self.items = {}

            def add(self, name, qty, price):
                if qty <= 0:
                    raise ValueError("quantidade deve ser positiva")
                q, _ = self.items.get(name, (0, price))
                self.items[name] = (q + qty, price)
        ''',
    "test_inventory.py": '''
        from inventory import Inventory

        inv = Inventory()
        inv.add("caneta", 10, 2.5)
        inv.add("caderno", 3, 12.0)
        inv.remove("caneta", 4)
        assert inv.items["caneta"] == (6, 2.5)
        assert abs(inv.total_value() - (6 * 2.5 + 3 * 12.0)) < 1e-9
        try:
            inv.remove("caderno", 5)
            raise SystemExit("remove deveria falhar sem estoque suficiente")
        except ValueError:
            pass
        inv.remove("caderno", 3)
        assert "caderno" not in inv.items
        print("OK")
        ''',
}, "Implemente em inventory.py os metodos remove(name, qty) e total_value() que os testes de test_inventory.py esperam. "
   "Rode python test_inventory.py ate passar.", '''
    import subprocess, sys
    r = subprocess.run([sys.executable, "test_inventory.py"], capture_output=True, text=True)
    sys.exit(0 if r.returncode == 0 and "OK" in r.stdout else 1)
    ''')

# 3. a rename across files, behaviour kept
task("refactor_rename", {
    "utils.py": '''
        def calc(nums):
            return sum(nums) / len(nums) if nums else 0.0
        ''',
    "report.py": '''
        from utils import calc


        def resumo(notas):
            return f"media {calc(notas):.2f} de {len(notas)} notas"
        ''',
    "main.py": '''
        from utils import calc
        from report import resumo

        if __name__ == "__main__":
            notas = [7.0, 8.5, 9.0]
            print(calc(notas))
            print(resumo(notas))
        ''',
    "test_report.py": '''
        import report
        import utils

        assert report.resumo([6, 8]) == "media 7.00 de 2 notas"
        assert utils.calcular_media([1, 2, 3]) == 2
        assert utils.calcular_media([]) == 0.0
        print("OK")
        ''',
}, "Renomeie a funcao calc de utils.py para calcular_media em todo o projeto (utils.py, report.py, main.py), "
   "sem mudar o comportamento. Depois rode python test_report.py e python main.py.", '''
    import pathlib, subprocess, sys
    r = subprocess.run([sys.executable, "test_report.py"], capture_output=True, text=True)
    m = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
    left = [p.name for p in pathlib.Path(".").glob("*.py") if "calc(" in p.read_text(encoding="utf-8") and p.name != "check.py"]
    sys.exit(0 if r.returncode == 0 and m.returncode == 0 and "OK" in r.stdout and not left else 1)
    ''')

# 4. a bug to find: quoted fields with commas
task("debug_csv", {
    "csvmini.py": '''
        def parse_line(line):
            """Campos de uma linha CSV: separados por virgula; um campo entre aspas pode ter virgulas,
            e "" dentro dele e uma aspa."""
            return [f.strip() for f in line.rstrip("\\n").split(",")]


        def parse(text):
            return [parse_line(l) for l in text.splitlines() if l.strip()]
        ''',
    "test_csvmini.py": '''
        from csvmini import parse, parse_line

        assert parse_line("a,b,c") == ["a", "b", "c"]
        assert parse_line('nome,"Silva, Jose",30') == ["nome", "Silva, Jose", "30"]
        assert parse_line('x,"diz ""oi""",y') == ["x", 'diz "oi"', "y"]
        assert parse("a,b\\n\\n1,2\\n") == [["a", "b"], ["1", "2"]]
        print("OK")
        ''',
}, "python test_csvmini.py falha. Descubra a causa e corrija csvmini.py sem usar o modulo csv da biblioteca padrao. "
   "Rode o teste ate passar.", '''
    import pathlib, subprocess, sys
    r = subprocess.run([sys.executable, "test_csvmini.py"], capture_output=True, text=True)
    src = pathlib.Path("csvmini.py").read_text(encoding="utf-8")
    sys.exit(0 if r.returncode == 0 and "OK" in r.stdout and "import csv" not in src else 1)
    ''')

# 5. a feature across the program and its interface
task("feature_cli_json", {
    "wordcount.py": '''
        import sys


        def count(text):
            words = text.split()
            return {"linhas": len(text.splitlines()), "palavras": len(words), "caracteres": len(text)}


        def main(argv):
            if len(argv) < 2:
                print("uso: python wordcount.py ARQUIVO")
                return 2
            with open(argv[1], encoding="utf-8") as f:
                c = count(f.read())
            print(f"{c['linhas']} linhas, {c['palavras']} palavras, {c['caracteres']} caracteres")
            return 0


        if __name__ == "__main__":
            sys.exit(main(sys.argv))
        ''',
    "dados.txt": "um dois tres\nquatro cinco\n",
}, "Adicione ao wordcount.py uma opcao --json: com ela, a saida e um objeto JSON com as chaves linhas, palavras e "
   "caracteres; sem ela, a saida continua igual. Escreva um teste test_wordcount.py que rode o programa nos dois modos "
   "com dados.txt e rode o teste.", '''
    import json, pathlib, subprocess, sys
    a = subprocess.run([sys.executable, "wordcount.py", "dados.txt", "--json"], capture_output=True, text=True)
    b = subprocess.run([sys.executable, "wordcount.py", "dados.txt"], capture_output=True, text=True)
    try:
        j = json.loads(a.stdout)
        ok = j == {"linhas": 2, "palavras": 5, "caracteres": 26}
    except Exception:
        ok = False
    ok = ok and b.stdout.strip() == "2 linhas, 5 palavras, 26 caracteres" and pathlib.Path("test_wordcount.py").exists()
    t = subprocess.run([sys.executable, "test_wordcount.py"], capture_output=True, text=True) if ok else None
    sys.exit(0 if ok and t.returncode == 0 else 1)
    ''')

print("tasks in", ROOT)
