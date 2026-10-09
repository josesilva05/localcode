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
   - 2b: contexto maior no motor.
     - **Feito para o Qwen3.6-35B-A3B (2026-10-08): 128k por padrão** (o teto é o `context_length` do GGUF, 262.144).
       - Atenção da GPU reescrita no estilo flash: blocos de posições com softmax incremental; nada mais guarda as notas de todas as posições na memória compartilhada (isso limitava a ~12k).
       - Acima de 32k posições, K/V em fp16 (`Q38_KV=16|32`; 128k custam 2,7 GB de VRAM) e atenção nos tensor cores (`mma.sync`, fp16 com acumulação fp32, a precisão do FlashAttention do llama.cpp):
         - decodificação: 128 segmentos de um warp por cabeça K/V, juntados depois;
         - prompt: 16 tokens × 8 cabeças por bloco, a saída nos registradores.
       - Até 32k, K/V e atenção continuam em fp32, exatos contra a referência (decodificação em 32 segmentos).
       - Especialistas de lotes grandes viram GEMMs do cuBLAS (pesos desquantizados uma vez para fp16; `Q38_XGEMM_MIN=0` desliga).
       - Validação contra float64: caminhos fp32 a 5e-7; modo fp16 ~1e-4, com o mesmo top-5. Flash-Next sem regressão (9,7e-7 contra a versão anterior).
       - Teste da agulha (uma senha no meio do texto, pergunta no fim): achada a 16k e a 118k tokens.
       - Medidas no 35B-A3B (pools de VRAM pré-carregados como no localcode):
         - prompt de 118,6k tokens em 501 s (237 tok/s; ~3,3 s por 1024 tokens no começo e 5,3 s perto de 115k);
         - decodificação a 118k de profundidade: ~13,5 tok/s; no início do contexto, 31–33 tok/s.
     - Falta: Flash-Next acima de 32k (ordenação do indexador limitada a 8192 blocos; continua com 8k por padrão).
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
- `/auto` (modo autônomo da pasta, lembrado em `%LOCALAPPDATA%\localcode\auto_dirs.txt`):
  - escritas e comandos rodam sem pedir aprovação;
  - cada pedido leva a regra de não fazer perguntas: o modelo escolhe a opção mais razoável, diz qual escolheu e vai até o fim;
  - o rodapé mostra "⏵⏵ autônomo nesta pasta"; `/auto off` desliga.
  - Os limites do ciclo continuam valendo (5 tentativas, a mesma falha 3 vezes) e as ferramentas continuam presas à pasta de trabalho.

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

### Os modelos desta máquina (2026-10-07)

| Modelo | Arquitetura | Estado | Velocidade (localcode, GPU) |
|---|---|---|---|
| Qwen3.8-Flash-Next UD-Q4_K_XL / UD-IQ4_XS | `qwen4exp` | validado (float64) | ~12 tok/s (limitado por RAM/disco) |
| Qwen3.6-35B-A3B-MTP | `qwen35moe` | validado (float64: CPU 3e-6, GPU 5e-7); contexto 128k | 32,8 tok/s; agente 31,2 tok/s; 13,5 tok/s a 118k |
| Qwen3.6-35B-A3B-Uncensored | `qwen35moe` | roda | 35,5 tok/s |
| Qwen3.8-27B UD-Q5_K_M | `qwen35` (denso) | roda na CPU e nas 2 GPUs; contexto 32k | 14,6 tok/s (llama.cpp: 15,4); prompt 435 tok/s (llama.cpp: 489); agente 14,5 tok/s |

**O 27B (2026-10-09).**
- Denso: uma FFN por camada (guardada nos campos do especialista compartilhado, sem o portão).
- As 64 camadas dividem-se entre as duas 3060 (19,4 GB de pesos): cada GPU é uma "parte" com suas camadas, K/V e estado; o resíduo do token passa de uma para a outra pelo host (`q38_gpu_dense.cuh`).
- Escrita: os pesos ficam nos tipos do GGUF e o produto matriz×vetor usa ativações em 8 bits e `dp4a` (o `mmvq` do llama.cpp, `q38_gpu_mvq.cuh`); `Q38_ACT8=0` volta ao caminho float exato.
- Prompt (`q38_gpu_dense_batch.cuh`):
  - em pedaços de 1024 tokens; cada matriz é desquantizada para fp16 e multiplicada no cuBLAS;
  - as duas GPUs trabalham em paralelo (a GPU 1 começa o pedaço seguinte enquanto a 0 faz o anterior);
  - a partir de 16 tokens o lote já ganha de token a token.
- A atenção nos tensor cores aceita grupos de 6 cabeças por cabeça K/V.
- O estado salvo (prefixos, `/resume`, `--continue`) cobre as duas GPUs.
- Contexto: 32k por padrão; o K/V em fp16 de 64k não cabe junto com os pesos nos 24 GB.

Depois: MTP (geração especulativa com a camada multi-token do 35B-MTP e do 27B).

## Próxima implementação: a API em Lua (pedido de 2026-10-07)

Reescrever a API (hoje `tools/q38_server.py`, Python sobre `q38api.dll`) em Lua integrada ao C, na linha do Kong (OpenResty/LuaJIT): o servidor HTTP e o roteamento em C ou LuaJIT, as rotas, o formato OpenAI/llama-server, os prefixos e o estacionamento de conversas em Lua, o motor chamado direto pela FFI do LuaJIT (sem processo Python no meio). Também candidato: perfis de modelo e ferramentas do agente escritos em Lua (plugins sem recompilar).
