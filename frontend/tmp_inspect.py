from pathlib import Path
lines = Path('index.html').read_text(encoding='utf-8').splitlines()
found = False
for idx, line in enumerate(lines, 1):
    if 'handleLoginSubmit' in line:
        found = True
        for j in range(max(1, idx - 10), min(len(lines) + 1, idx + 20)):
            print(f'{j}: {lines[j-1]}')
        break
if not found:
    print('not found')
