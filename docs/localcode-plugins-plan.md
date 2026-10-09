# localcode: skills, plugins e mods

Pedido de 2026-10-09: as skills do guia "As 50 melhores Claude Skills" (Asimov Academy) viram as
skills do sistema do localcode, a arquitetura de mods do `anthropics/claude-code-playground` funciona
de verdade, e depois disso tudo vira plugin.

## O que foi baixado

Só as de licença livre (MIT, Apache-2.0, CC0, CC-BY-SA com atribuição), clone raso de 24 repositórios
em `%LOCALAPPDATA%\localcode\skills\vendor\<dono>__<repo>` (478 MB, fora do git). Nada deles é
executado ao baixar.

- 35 dos 50 itens do guia.
- Ficaram de fora 15:
  - sem licença: vercel-labs/agent-skills (19, 20, 21), karpathy-guidelines (33), firecrawl (41), remotion (44) e doc-coauthoring (38);
  - de documento da Anthropic, todos os direitos reservados: docx, pdf, pptx e xlsx (34-37, 49); foram apagadas do clone;
  - o Claude Art Skill (47) não existe mais (404), e o Impeccable (23) é só um site;
  - o pacote "Document Skills" (50): as partes livres já entram pelos outros itens.
- Nos 24 repositórios: 556 skills, 234 comandos, 231 agentes e 25 arquivos de hooks.

## Os formatos

**Plugin do Claude Code.** Uma pasta com `.claude-plugin/plugin.json` (name, version, description,
author...). Componentes, nos caminhos padrão ou nos que o manifesto der:
- `skills/<nome>/SKILL.md`: cabeçalho YAML (`name`, `description`, às vezes em várias linhas com `|`
  ou `>`), depois as instruções; outros arquivos da pasta são lidos sob demanda;
- `commands/*.md`: comandos `/nome`; o texto vira a mensagem, com `$ARGUMENTS`, `$1`...;
- `agents/*.md`: subagentes (cabeçalho com name, description, tools, model);
- `hooks/hooks.json` ou `"hooks"` no manifesto: eventos `SessionStart`, `UserPromptSubmit`,
  `PreToolUse`, `PostToolUse`, `Stop`, `SubagentStart`, `PreCompact`, `SessionEnd`, com `matcher` no
  nome da ferramenta, e hooks `{"type": "command", "command", "args", "timeout"}`;
- `.claude-plugin/marketplace.json`: uma coleção de plugins, com `source` relativo.

**Mod** (claude-code-playground): um plugin cujo hook é um módulo JavaScript (`hooks.json` →
`{"modules": ["./x.mjs"]}`) que exporta `register(on)`:
- eventos: `session.start`, `turn.start`, `turn.complete`, `tool.call` (intermediário: pode segurar,
  mudar ou cancelar a chamada), `command.run`, `ui.render` (`AbovePrompt`, `Pane`) e `ui.close`;
- host `$`: `process.run`, `fs.read/exists`, `clock.now/sleep`, `session.cwd/usage`,
  `ui.open/close/invalidate/toast/resolve` (Box, Text, Button) e `command.register`.

## Como entra no localcode

Fase 1, carregador de plugins e skills (`src/cli/lc_plugins.c`):
- Raízes:
  - do sistema: `%LOCALAPPDATA%\localcode\skills\vendor` e `%LOCALAPPDATA%\localcode\plugins`;
  - do usuário: `%USERPROFILE%\.localcode\plugins` e `...\skills`;
  - do projeto: `.localcode/plugins`, `.localcode/skills` e `.claude/skills`.
- Em cada pasta: plugin (`plugin.json`), coleção (`marketplace.json`) ou só skills.
- Ligado/desligado por plugin e por skill em `%LOCALAPPDATA%\localcode\plugins.txt`; por padrão, os 35
  itens do guia.
- O prompt do sistema lista as skills ligadas dentro de um orçamento de tokens (o modelo é local, e
  556 descrições passariam de 20 mil tokens).
- `skill_search` acha as outras por palavra; `skill` aceita `plugin:nome` quando dois plugins têm a
  mesma skill.
- `/plugin` lista, liga, desliga e mostra.

Fase 2, comandos dos plugins como `/nome` no menu (com `$ARGUMENTS`).

Fase 3, hooks do Claude Code:
- O localcode roda os comandos com o JSON do evento na entrada e `CLAUDE_PLUGIN_ROOT` e
  `CLAUDE_PROJECT_DIR` no ambiente.
- Código de saída 2 bloqueia, e o stderr volta para o modelo.
- Do JSON da saída valem `decision`/`permissionDecision`, `additionalContext` e `systemMessage`.
- Nomes de ferramenta do Claude Code para os do localcode: Read=read_file, Write=write_file,
  Edit=edit_file, Bash=run, Glob=glob, Grep=grep, LS=list_dir, Skill=skill.

Fase 4, mods:
- Um host em Node (`localcode-modhost.mjs`; Node 24 já está nesta máquina) carrega os módulos e fala
  com o localcode por linhas JSON.
- `tool.call` segura a ferramenta até o mod responder.
- `ui.render` devolve a árvore Box/Text/Button, que a TUI desenha acima do prompt ou num painel com
  teclas.
- Teste: os três mods do playground sem mudar uma linha (`blast-radius`, `replay-theater` e
  `token-weather`).

Fase 5, subagentes dos plugins (`agents/*.md`):
- Uma conversa à parte com o mesmo modelo, que devolve um resumo.
- O estado da conversa principal fica salvo e é retomado depois.

Fase 6:
- Testar cada um dos 35 itens com o Qwen local: ativa quando deve, muda o comportamento, não quebra.
- Depois, as partes próprias do localcode também viram plugins.

## Estado (2026-10-09)

- **Fase 1, feita (f013137).** O carregador acha 73 plugins, 446 skills e 161 comandos em 0,4 s.
  - O prompt lista as skills ligadas dentro de 7000 caracteres, com as do guia primeiro; `skill_search` acha as outras.
  - O 27B achou as skills de auditoria de dependências pela busca e carregou `test-driven-development` numa tarefa de TDD.
  - Falta: com a skill carregada, ele ainda não roda o teste novo para vê-lo falhar antes de implementar.
- **Fase 2, feita (0484b6e).** `/nome` roda os comandos dos plugins e as skills chamadas pelo usuário, com o prefixo curto do plugin quando há conflito. O `/caveman-review` funcionou com o agente.
- **Fase 3, feita (bb6ca68).** Hooks de comando do Claude Code, só depois de `/plugin hooks on NOME`, e os do projeto (`.claude/settings.json`).
  - Com o 27B, funcionaram UserPromptSubmit, PreToolUse (bloqueou `rm`), PostToolUse e Stop.
  - O SessionStart do superpowers funcionou.
- **Fase 4, feita.** O `modhost.mjs` em Node roda os três mods do playground sem mudar uma linha.
  - Blast Radius segurou `git clean -f velho.bak`, mediu "remove 1 untracked path", desenhou o painel, e o Cancel recusou o comando.
  - Replay Theater registrou a edição e respondeu ao `/replay`.
  - Token Weather desenhou a faixa acima do prompt na TUI.
  - Painel na TUI: Tab e as setas movem o foco, Enter e as teclas de atalho apertam, Esc fecha ou interrompe. Na faixa, Alt+tecla aperta. No modo texto, o painel é impresso e respondido por uma linha.
- **Incompatibilidade.** A ferramenta `run` do localcode usa PowerShell, mas skills, hooks e mods do ecossistema do Claude Code supõem bash.
  - Exemplo: o Blast Radius não reconhece `Remove-Item -Recurse -Force`.
  - A decidir: oferecer a bash (Git Bash, como o Claude Code no Windows) como shell do `run`.
- **Fases 5 e 6, por fazer:** subagentes dos plugins; testar os 35 itens com o Qwen local; tudo como plugin.
