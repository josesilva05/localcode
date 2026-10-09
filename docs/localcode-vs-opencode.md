# localcode contra o opencode v2

Pedido de 2026-10-09: o localcode melhor que o opencode (github.com/anomalyco/opencode, branch v2). Sem versão para celular, a pedido do usuário. O `run` usa a bash por padrão, também a pedido dele.

Fontes: o README da v2, `specs/v2` e a documentação em opencode.ai/docs (ferramentas, agentes, permissões,
LSP, MCP, servidor e plugins).

## Comparação

| Área | opencode v2 | localcode hoje |
|---|---|---|
| Modelos | Mais de 75 provedores por API (nuvem ou local) | Motor próprio, local: Qwen3.8-27B a 53 tok/s com MTP e paralelismo de tensor, 35B-A3B a 33, Flash-Next; troca com `/model` |
| Arquitetura | Servidor HTTP + SSE (OpenAPI, SDK); a TUI é um cliente; IDE, web e desktop | A TUI e o agente num programa só; a API em Python (`q38_server.py`) à parte |
| Agentes | Build e Plan (Tab troca); subagentes general, explore e scout pela ferramenta Task ou por @menção, em sessões filhas | Níveis 0, 1 e 2 (chat, só leitura, agente); sem subagentes; os 231 agentes dos plugins sem uso |
| Ferramentas | bash, edit, write, read, grep (regex), glob (por data), apply_patch, skill, todowrite, webfetch, websearch, question, lsp | read_file, list_dir, glob, grep (texto literal), write_file, edit_file, run (PowerShell), plan, test, skill, skill_search |
| Permissões | allow/ask/deny por ferramenta e por padrão (`"git status *"`), a última regra que casa vale; `.env` bloqueado; external_directory; doom_loop; "sempre" com o padrão sugerido; `--auto` | Pergunta S/A/N por tipo de ferramenta, `/approve auto` e `/auto` por pasta |
| Regras | `AGENTS.md` e `/init` | Nenhuma |
| Desfazer | `/undo` e `/redo` | Nenhum |
| Skills | `SKILL.md` | 446 skills (35 itens do guia), busca, `/skill` |
| Plugins | Os próprios, em JS/TS (tool.execute.before/after, eventos de sessão, ferramentas novas, shell.env) | Os do Claude Code: skills, comandos, hooks e mods em Node |
| MCP | Servidores locais e remotos, com OAuth | Nenhum |
| LSP e formatadores | Diagnósticos (opcionais) e formatar depois de editar | Nenhum |
| TUI | @arquivo com busca aproximada, imagens, paleta de comandos, temas, atalhos configuráveis, navegar entre sessões filhas | Menu `/`, aprovação, faixa e painel dos mods, `/resume`, compactação |
| Compartilhar, GitHub e GitLab | `/share` (link); o agente no CI | Nenhum |
| Laço de engenharia | O do modelo | Plano com teste obrigatório, testar depois de editar, limite de tentativas |
| Privacidade e custo | Depende do provedor | Tudo local, sem custo por token |

## Onde o localcode já passa à frente

- **Motor próprio e local:**
  - afinado para estas placas;
  - sem nuvem e sem custo por token;
  - privacidade total, o que conta para o harness corporativo.
- **Ecossistema do Claude Code inteiro:** plugins, skills, comandos, hooks e mods. O opencode lê só as skills e tem um sistema de plugins próprio.
- **Laço de engenharia guiado:** plano, teste depois de cada mudança, limite de tentativas.

## Roteiro para passar à frente

**Onda 1: paridade do agente.** Rápida e de alto impacto.
1. Regras do projeto: `AGENTS.md`, e `CLAUDE.md` também; `/init` para criar.
2. `/undo` e `/redo`: um instantâneo dos arquivos que cada turno tocou, sem depender do git.
3. Ferramentas:
   - grep com regex e glob por data;
   - `apply_patch`;
   - `webfetch`;
   - `question` (perguntar com opções);
   - lista de tarefas.
4. Shell do `run`: bash (Git Bash) para casar com o ecossistema, com o PowerShell como opção.
5. Permissões granulares no formato do opencode (allow/ask/deny por padrão), com:
   - `.env` bloqueado;
   - aviso ao sair da pasta;
   - detecção de laço.
6. TUI: @arquivo com busca aproximada; Tab troca entre Build e Plan.

**Onda 2: arquitetura.**
1. Subagentes:
   - a ferramenta Task e a @menção;
   - os `agents/*.md` dos plugins;
   - cada um num contexto à parte no mesmo motor (estado salvo e retomado).
2. MCP: servidores locais (stdio) e remotos (HTTP).
3. LSP: diagnósticos depois de editar; formatadores.
4. `localcode serve`: HTTP + SSE com sessões, mensagens e eventos.
   - a TUI vira um cliente;
   - IDE e web podem conectar;
   - as máquinas da rede usam o mesmo servidor, que se junta com a API corporativa em Lua (auth, fila, vários modelos).

**Onda 3: medir.** "Melhor" precisa de número.
- Um conjunto de tarefas reais: corrigir, implementar, refatorar e testar em projetos pequenos.
- Rodado pelo localcode e pelo opencode com o **mesmo modelo**: o opencode fala com o nosso motor pela API compatível com a OpenAI.
- Medidas: tarefas resolvidas, tempo e tokens.
- A cada onda, a tabela de resultados mostra se passamos à frente.
