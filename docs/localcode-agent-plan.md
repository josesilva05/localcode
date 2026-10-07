# Agente do localcode — 3 níveis, o último com gauntlet-loop

Plano de 2026-10-05. O agente roda dentro do `localcode` (C), sobre o mesmo motor em processo, sem o harness.

## Os 3 níveis (`/nivel 1|2|3`)

| Nível | Para quê | Raciocínio | Ferramentas | Duração típica |
|---|---|---|---|---|
| 1 Rápido | perguntas, explicar código | low | só leitura (ler, listar, buscar) | segundos |
| 2 Agente | implementar, corrigir, refatorar | medium | todas, com aprovação de escrita e comandos | minutos |
| 3 Gauntlet | entregas que precisam ser excelentes | xhigh no crítico | todas + crítico + comparação cega | até 2 h |

## Nível 3 — gauntlet-loop (Matt Shumer)

1. **Barra de qualidade**: antes de começar, o agente propõe 2–3 referências concretas (nomeada, que dá para abrir/rodar, comparável lado a lado) e você escolhe uma.
   - código: suíte de testes + especificação, ou um programa de referência para comparar saídas;
   - web/jogos: uma página ou app de referência; o crítico usa Playwright (DOM, console, erros, interação roteirizada) — sem screenshots, porque o modelo não tem visão;
   - documentos: um documento de referência.
2. **Decomposição**: o orquestrador (código C, determinístico) pede ao modelo uma lista de peças pequenas, cada uma melhorável e julgável sozinha.
3. **Loop por peça**: construtor (contexto limpo) → crítico (contexto limpo, nunca vê a justificativa do construtor; roda testes, comandos e Playwright de verdade) → **comparação cega**: o crítico recebe A e B sem rótulos (nosso resultado e a referência, em ordem sorteada) e escolhe o melhor, sem nota numérica → se perdermos, os motivos voltam ao construtor e a peça roda de novo.
4. **Parada**: quando vence a comparação, quando você interrompe, ou ao fim do orçamento (2 h) — aí mostra o melhor resultado e o que faltou.

Restrições do nosso hardware: um modelo só a ~10 tok/s → os subagentes rodam um por vez (não em paralelo); cada um tem contexto próprio guardado e restaurado por `q38api_state_save/load` (0,1–0,5 s por troca), e o prompt de sistema de cada papel fica salvo em arquivo como no servidor.

## Arquitetura (C)

- `src/cli/lc_agent.c/.h` — loop do agente: monta o prompt com o template do Qwen e o bloco `# Tools`, gera, interpreta as chamadas de ferramenta, executa, devolve o resultado, repete.
- `src/cli/lc_tools.c/.h` — ferramentas: `read_file` (faixa de linhas), `write_file`, `edit_file` (troca de texto exato), `list_dir`, `glob`, `grep`, `run` (pwsh com timeout e limite de saída), `todo`, `ask_user`. Tudo confinado à pasta de trabalho; modos de permissão: só leitura / escrita na pasta / acesso total.
- `src/cli/lc_session.c` — sessão em JSONL (`%LOCALAPPDATA%\localcode\sessions`) e compactação por resumo perto do fim do contexto, qualquer que seja o tamanho (32k, 64k, 128k, 256k): o modelo escreve um resumo para si mesmo (até contexto/16 tokens, 768–8192), o contexto é esvaziado e o resumo abre o próximo; repete quantas vezes precisar. Ao sair, o estado do motor fica guardado ao lado (as 3 últimas conversas) e `/resume` / `--continue` volta exatamente; conversas mais antigas voltam pelo texto (último resumo + mensagens depois dele).
- `src/cli/lc_gauntlet.c` — orquestrador do nível 3 (barra, peças, construtor, crítico, comparação cega, orçamento).
- `tools/lc_web_probe.mjs` — sonda Playwright usada pelo crítico de web.
- Engine vtable: `save_state` / `load_state` para os contextos dos subagentes.

## Fases (cada uma medida e validada antes da próxima)

1. Ferramentas + loop do agente + níveis 1 e 2 no TUI (chamadas de ferramenta visíveis, aprovação).
2. Sessões em disco + compactação.
   - 2a: `lc_session.c`, `/compact [foco]`, `/sessions`, `/resume [N]`, `--continue`; limite proporcional ao contexto.
   - 2b: contexto maior no motor — hoje 32k na GPU (limite da ordenação do indexador) e 64k na CPU; o modelo vai a 262.144. KV em fp32 custa ~54 KB/posição (12 camadas de atenção, 2 cabeças KV × 256, mais o indexador): 1,8 GB em 32k, 7 GB em 128k, 14 GB em 256k. Caminho: KV em fp16 (metade), ordenação do indexador sem o teto de 8192 blocos, e medir VRAM tirada do cache de especialistas × velocidade. Preferido: 128k; topo: 256k.
3. Contextos de subagente + nível 3 para código com testes.
4. Críticos de web (Playwright) e de documentos.
5. Ajuste com tarefas reais: tempo, taxa de vitória na comparação cega, regressões.

## O ciclo de programação do nível 2 (2026-10-07)

Planejar → programar → testar → (depurar → corrigir → retestar)* → finalizar, conduzido pelo código (`lc_agent.c`, `Flow`), não pela palavra do modelo:

- `plan(steps, test_command)` registra os passos e o comando que prova a tarefa (código de saída 0 = funciona); a tela mostra o plano numerado.
- `test([command])` roda o comando; o código de saída decide. Uma falha **depois de uma mudança nova** é uma tentativa; no máximo **5** tentativas, ou **a mesma falha 3 vezes seguidas**: aí o controlador manda parar e responder o que tentou e o que falta, e não executa mais ferramentas.
- Passou: o agente finaliza na hora (resposta curta: o que mudou e o teste que passou). Se passar de primeira, acabou.
- Uma resposta final depois de mudar arquivos sem testar é devolvida para testar (no máximo 2 vezes).
- Uma pergunta que não muda nada é respondida direto, sem plano nem teste. O agente segue direto, sem esperar aprovação do plano.

**Skills**: pastas com `SKILL.md` (cabeçalho `name:` / `description:` entre `---`, depois as instruções) em `.localcode/skills` e `.claude/skills` da pasta de trabalho e em `%USERPROFILE%\.localcode\skills`. O agente vê nomes e descrições; `skill(name)` carrega as instruções (e `file` outros arquivos da pasta da skill). Quando uma skill cobre a tarefa, ela vale sobre o ciclo padrão.

## Um motor que se ajusta a cada modelo (desenho, 2026-10-07)

Sem clonar o projeto por modelo. Três camadas:

| Camada | O quê | Onde |
|---|---|---|
| Núcleo comum | leitor GGUF, todos os tipos de quantização (CPU e GPU), cache de especialistas, monitor de disco, arquivos de estado, tokenizador, interface e agente | uma biblioteca (`lc_core`) |
| Motor da família | a arquitetura (Qwen3.8: GDN + atenção com indexador + PLE + hyper-connections; GLM-5.3: MLA + DSA) | um módulo por família, escolhido por `general.architecture` do GGUF; compilados juntos, com registro |
| Perfil do modelo | formato de chamada de ferramenta, marcas de raciocínio, níveis de esforço, contexto padrão, orçamento de compactação, ajustes de prompt (pt-BR), limites do agente, parâmetros de cache/VRAM | `perfil.json` por modelo/variante, editável sem recompilar |

Calibração (`localcode --calibrar`): mede nesta máquina e grava no perfil — velocidade (cache de RAM, divisão de VRAM, lote, prefetch, sem passar de 95% de disco) e qualidade (um conjunto pequeno de tarefas de agente: acertos, passos, tempo).

Ordem: (1) o ciclo do agente — feito; (2) perfil de modelo; (3) núcleo comum + registro por arquitetura (juntar o que está duplicado entre os motores GLM e Qwen); (4) calibração.
