# 🧠 Paradigm-Shift-in-Agentic-AI-Papers

<div align="center">
  <img src="logo.png" alt="Logo" width="300">
  <h1 align="center">Agentic AI: From Pipelines to Model-native</h1>
  <p align="center">
    This repository contains a curated list of papers referenced in our survey: <br>
    <a href="https://github.com/ADaM-BJTU/Paradigm-Shift-in-Agentic-AI-Papers"><strong>A Paradigm Shift in Agentic AI: Model-native Planning, Tool Use, and Memory beyond Pipelines</strong></a>
  </p>
  
  [![Awesome](https://awesome.re/badge.svg)](https://github.com/ADaM-BJTU/Paradigm-Shift-in-Agentic-AI-Papers) 
  ![](https://img.shields.io/github/last-commit/ADaM-BJTU/Paradigm-Shift-in-Agentic-AI-Papers?color=green) 

</div>


## 🙏 Citation

If you find our survey useful for your research, please consider citing our work:

```bibtex
@article{
}
```


## 📒 Table of Contents
- [Core Capabilities: Planning](#-3core-capabilities-planning)
    - [Pipeline-based Paradigm](#32pipeline-based-paradigm)
    - [Model-native Paradigm](#33model-native-paradigm)
- [Core Capabilities: Tool Use](#-4core-capabilities-tool-use)
    - [Pipeline-based Paradigm](#42pipeline-based-paradigm)
    - [Model-native Paradigm](#43model-native-paradigm)
- [Core Capabilities: Memory](#-5core-capabilities-memory)
- [Applications](#-6applications)
    - [Deep Research Agent](#61deep-research-agent)
    - [GUI Agent](#62gui-agent)
- [Future Direction and Discussion: Emerging Model-native Capabilities](#-7future-direction-and-discussion)
    - [Multi-agent Collaboration](#711emerging-model-native-capabilities-multi-agent-collaboration)
    - [Pipeline-based paradigm](#712emerging-model-native-capabilities-pipeline-based-paradigm)


# 📜 Papers

## ➤ 3&nbsp;&nbsp;Core Capabilities: Planning

### 3.2&nbsp;&nbsp;Pipeline-based Paradigm

|  Short Name  |   Paper   |   Date  |  Code/Project Link |
|  :---------: |   :---:   |   :--:  |  :---------------: |
|   STRIPS  |  [Strips: A new approach to the application of theorem proving to problem solving](https://www.sciencedirect.com/science/article/pii/0004370271900105)  |  1971  | - |
|   PDDL  |  [PDDL - The Planning Domain Definition Language](https://www.cs.cmu.edu/~mmv/planning/readings/98aips-PDDL.pdf)  |  1998  | - |
|   LLM+P  |  [{LLM+P: Empowering Large Language Models with Optimal Planning Proficiency](https://arxiv.org/abs/2304.11477)  |  2023-04  | [GitHub](https://github.com/Cranial-XIX/llm-pddl) ![Stars](https://img.shields.io/github/stars/Cranial-XIX/llm-pddl) |
|   LLM+PDDL  |  [Leveraging pre-trained large language models to construct and utilize world models for model-based task planning](https://arxiv.org/abs/2305.14909)  |  2023-05  | [GitHub](https://github.com/GuanSuns/LLMs-World-Models-for-Planning) ![Stars](https://img.shields.io/github/stars/GuanSuns/LLMs-World-Models-for-Planning) |
|   CoT  |  [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)  |  2022-01  | - |
|   ToT  |  [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/pdf/2305.10601)  |  2023-05  | [GitHub](https://github.com/princeton-nlp/tree-of-thought-llm) ![Stars](https://img.shields.io/github/stars/princeton-nlp/tree-of-thought-llm) |
|   RAP  |  [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992)  |  2023-05  | - |
|   LLM+MCTS  |  [Planning with MCTS: Enhancing Problem-Solving in Large Language Models](https://openreview.net/forum?id=sdpVfWOUQA)  |  2024-09  | - |

### 3.3&nbsp;&nbsp;Model-native Paradigm

|  Short Name  |   Paper   |   Date  |  Code/Project Link |
|  :---------: |   :---:   |   :--:  |  :---------------: |
|   ReST-MCTS*  |  [ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search](https://arxiv.org/abs/2406.03816)  |  2024-06  | [GitHub](https://github.com/THUDM/ReST-MCTS) ![Stars](https://img.shields.io/github/stars/THUDM/ReST-MCTS) |
|   Marco-o1  |  [Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions](https://arxiv.org/abs/2411.14405)  |  2024-11  | [GitHub](https://github.com/AIDC-AI/Marco-o1) ![Stars](https://img.shields.io/github/stars/AIDC-AI/Marco-o1) |
|   HuatuoGPT-o1  |  [HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs](https://arxiv.org/abs/2412.18925)  |  2024-12  | [GitHub](https://github.com/FreedomIntelligence/HuatuoGPT-o1) ![Stars](https://img.shields.io/github/stars/FreedomIntelligence/HuatuoGPT-o1) |
|   Bespoke-Stratos  |  [Bespoke-Stratos: The unreasonable effectiveness of reasoning distillation](https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation)  |  2024-12  | [Dataset](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) |
|   s1  |  [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)  |  2025-01  | [GitHub](https://github.com/SimpleScaling/s1) ![Stars](https://img.shields.io/github/stars/SimpleScaling/s1) |
|   R1-Distill-SFT  |  [Millions scale dataset distilled from R1-32b](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT)  |  2025-01  | [Dataset](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT) |
|   LIMO  |  [LIMO: Less is More for Reasoning](https://arxiv.org/abs/2502.03387)  |  2025-02  | [GitHub](https://github.com/GAIR-NLP/LIMO) ![Stars](https://img.shields.io/github/stars/GAIR-NLP/LIMO) |
|   BOLT  |  [BOLT: Bootstrap Long Chain-of-Thought in Language Models without Distillation](https://arxiv.org/abs/2502.03860)  |  2025-02  | - |
|   AStar  |  [Boosting Multimodal Reasoning with Automated Structured Thinking](https://arxiv.org/abs/2502.02339)  |  2025-02  | - |
|   FastMCTS  |  [FastMCTS: A Simple Sampling Strategy for Data Synthesis](https://arxiv.org/abs/2502.11476)  |  2025-02  | - |
|   OpenThoughts  |  [OpenThoughts: Data Recipes for Reasoning Models](https://arxiv.org/abs/2506.04178)  |  2025-06  | [GitHub](https://github.com/open-thoughts/open-thoughts) ![Stars](https://img.shields.io/github/stars/open-thoughts/open-thoughts) |
|   OpenR1-Math-220k  |  [Open R1: A fully open reproduction of DeepSeek-R1](https://github.com/huggingface/open-r1)  |  2025-02  | [GitHub](https://github.com/huggingface/open-r1) ![Stars](https://img.shields.io/github/stars/huggingface/open-r1) |
|   SYNTHETIC-1   |  [SYNTHETIC-1: Two Million Collaboratively Generated Reasoning Traces from Deepseek-R1](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1-SFT-Data)  |  2025-02  | [Dataset](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1-SFT-Data) |
|   WebSynthesis  |  [WebSynthesis: World-Model-Guided MCTS for Efficient WebUI-Trajectory Synthesis](https://arxiv.org/abs/2507.04370)  |  2025-07  | [GitHub](https://github.com/LucusFigoGao/WebSynthesis) ![Stars](https://img.shields.io/github/stars/LucusFigoGao/WebSynthesis) |
|   Math-Shepherd  |  [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935)  |  2023-12  | [Project](https://achieved-bellflower-4d6.notion.site/Math-Shepherd-Verify-and-Reinforce-LLMs-Step-by-step-without-Human-Annotations-41b6e73c860840e08697d347f8889bac) |
|   ReFT  |  [ReFT: Reasoning with Reinforced Fine-Tuning](https://aclanthology.org/2024.acl-long.410/)  |  2024-01  | [GitHub](https://github.com/lqtrung1998/mwp_ReFT) ![Stars](https://img.shields.io/github/stars/lqtrung1998/mwp_ReFT) |
|   OmegaPRM  |  [Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/abs/2406.06592)  |  2024-06  | - |
|   OpenAI o1  |  [Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)  |  2024-09  | - |
|   RLEF  |  [RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning](https://arxiv.org/abs/2410.02089)  |  2024-10  | - |
|   o1-coder  |  [o1-Coder: an o1 Replication for Coding](https://arxiv.org/abs/2412.00154)  |  2024-01  | - |
|   Implicit PRM  |  [Free Process Rewards without Process Labels](https://raw.githubusercontent.com/mlresearch/v267/main/assets/yuan25c/yuan25c.pdf)  |  2024-12  | - |
|   ORPS  |  [Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation](https://api.semanticscholar.org/CorpusID:274859836)  |  2024-12  | [GitHub](https://github.com/zhuohaoyu/ORPS) ![Stars](https://img.shields.io/github/stars/zhuohaoyu/ORPS) |
|   OpenRFT  |  [OpenRFT: Adapting Reasoning Foundation Model for Domain-specific Tasks with Reinforcement Fine-Tuning](https://arxiv.org/abs/2412.16849)  |  2024-12  | [GitHub](https://github.com/ADaM-BJTU/OpenRFT) ![Stars](https://img.shields.io/github/stars/ADaM-BJTU/OpenRFT) |
|   DeepSeek R1  |  [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)  |  2025-01  | [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) ![Stars](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-R1) |
|   Qwen-2.5-Math-PRM  |  [The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B)  |  2025-01  | [Model](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B) |
|   Kimi k1.5  |  [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)  |  2025-01  | - |
|   O1-Pruner  |  [O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning](https://arxiv.org/abs/2501.12570)  |  2025-01  | [GitHub](https://github.com/StarDewXXX/O1-Pruner) ![Stars](https://img.shields.io/github/stars/StarDewXXX/O1-Pruner) |
|   PRIME  |  [Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456)  |  2025-02  | - |
|   DeepScaleR  |  [DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)  |  2025-02  | [GitHub](https://github.com/rllm-org/rllm) ![Stars](https://img.shields.io/github/stars/rllm-org/rllm) |
|   PRLCoder  |  [Process-Supervised Reinforcement Learning for Code Generation](https://arxiv.org/abs/2502.01715)  |  2025-02  | - |
|   L1  |  [L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning](https://arxiv.org/abs/2503.04697)  |  2025.03  | [GitHub](https://github.com/cmu-l3/l1) ![Stars](https://img.shields.io/github/stars/cmu-l3/l1) |
|   DAST  |  [DAST: Difficulty-Adaptive Slow-Thinking for Large Reasoning Models](https://arxiv.org/abs/2503.04472)  |  2025.03  | [GitHub](https://github.com/AnonymousUser0520/AnonymousRepo01) ![Stars](https://img.shields.io/github/stars/AnonymousUser0520/AnonymousRepo01) |
|   QwQ  |  [QwQ-32B: Embracing the Power of Reinforcement Learning](https://qwenlm.github.io/blog/qwq-32b/)  |  2025-03  | - |
|   Skywork or1  |  [Skywork Open Reasoner Series](https://capricious-hydrogen-41c.notion.site/Skywork-Open-Reaonser-Series-1d0bc9ae823a80459b46c149e4f51680)  |  2025-03  | - |
|   Demystify-long-cot  |  [Demystifying Long Chain-of-Thought Reasoning in LLMs](https://arxiv.org/abs/2502.03373)  |  2025-02  | [GitHub](https://github.com/eddycmu/demystify-long-cot) ![Stars](https://img.shields.io/github/stars/eddycmu/demystify-long-cot) |
|   LLM-as-Judge  |  [Who's Your Judge? On the Detectability of LLM-Generated Judgments](https://arxiv.org/abs/2509.25154)  |  2025-09  | - |

## ➤ 4&nbsp;&nbsp;Core Capabilities: Tool Use

### 4.2&nbsp;&nbsp;Pipeline-based Paradigm

|  Short Name  |   Paper   |   Date  |  Code/Project Link |
|  :---------: |   :---:   |   :--:  |  :---------------: |
|   WebGPT  |  [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332)  |  2021-12  | - |
|   SayCan  |  [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)  |  2022-04  | [GitHub](https://github.com/google-research/google-research/tree/master/saycan) ![Stars](https://img.shields.io/github/stars/google-research/google-research) 
|   WebShop  |  [WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://proceedings.neurips.cc/paper_files/paper/2022/hash/82ad13ec01f9fe44c01cb91814fd7b8c-Paper-Conference.pdf)  |  2022-07  | [GitHub](https://github.com/princeton-nlp/WebShop) ![Stars](https://img.shields.io/github/stars/princeton-nlp/WebShop) |
|   Code as Policies  |  [Code as Policies: Language Model Programs for Embodied Control](https://arxiv.org/abs/2209.07753)  |  2022-09  | [GitHub](https://github.com/google-research/google-research/tree/master/code_as_policies) ![Stars](https://img.shields.io/github/stars/google-research/google-research) |
|   HuggingGPT  |  [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face](https://proceedings.neurips.cc/paper_files/paper/2023/file/77c33e6a367922d003ff102ffb92b658-Paper-Conference.pdf)  |  2023-03  | [GitHub](https://github.com/microsoft/JARVIS) ![Stars](https://img.shields.io/github/stars/microsoft/JARVIS) |
|   AutoGen  |  [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)  |  2023-08  | [GitHub](https://github.com/microsoft/autogen) ![Stars](https://img.shields.io/github/stars/microsoft/autogen) |
|   SWE-agent  |  [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://papers.nips.cc/paper_files/paper/2024/file/5a7c947568c1b1328ccc5230172e1e7c-Paper-Conference.pdf)  |  2024-05  | [GitHub](https://github.com/SWE-agent/SWE-agent) ![Stars](https://img.shields.io/github/stars/SWE-agent/SWE-agent) |
|   Self-Ask  |  [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/abs/2210.03350)  |  2022-10  | [GitHub](https://github.com/ofirpress/self-ask) ![Stars](https://img.shields.io/github/stars/ofirpress/self-ask) |
|   ReAct  |  [ReAct: Synergizing Reasoning and Acting in Language Models](https://openreview.net/forum?id=WE_vluYUL-X)  |  2022-10  | [GitHub](https://github.com/ysymyth/ReAct) ![Stars](https://img.shields.io/github/stars/ysymyth/ReAct) |
|   PAL  |  [PAL: Program-aided Language Models](https://openreview.net/forum?id=WE_vluYUL-X)  |  2022-11  | [GitHub](https://github.com/reasoning-machines/pal) ![Stars](https://img.shields.io/github/stars/reasoning-machines/pal) |
|   PoT  |  [Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks](https://openreview.net/forum?id=YfZ4ZPt8zd)  |  2022-11  | [GitHub](https://github.com/TIGER-AI-Lab/Program-of-Thoughts) ![Stars](https://img.shields.io/github/stars/TIGER-AI-Lab/Program-of-Thoughts) |
|   Reflexion  |  [Reflexion: language agents with verbal reinforcement learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf)  |  2023-03  | [GitHub](https://github.com/NoahShinn024/reflexion) ![Stars](https://img.shields.io/github/stars/NoahShinn024/reflexion) |
|   ViperGPT  |  [ViperGPT: Visual Inference via Python Execution for Reasoning](https://openaccess.thecvf.com/content/ICCV2023/html/Suris_ViperGPT_Visual_Inference_via_Python_Execution_for_Reasoning_ICCV_2023_paper.html)  |  2023-03  | [GitHub](https://github.com/cvlab-columbia/viper) ![Stars](https://img.shields.io/github/stars/cvlab-columbia/viper) |
|   CRITIC  |  [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://openreview.net/forum?id=Sx038qxjek)  |  2023-05  | [GitHub](https://github.com/microsoft/ProphetNet/tree/master/CRITIC) ![Stars](https://img.shields.io/github/stars/microsoft/ProphetNet) |

### 4.3&nbsp;&nbsp;Model-native Paradigm

## ➤ 5&nbsp;&nbsp;Core Capabilities: Memory

## ➤ 6&nbsp;&nbsp;Applications

### 6.1&nbsp;&nbsp;Deep Research Agent

### 6.2&nbsp;&nbsp;GUI Agent

## ➤ 7&nbsp;&nbsp;Future Direction and Discussion

### 7.1.1&nbsp;&nbsp;Emerging Model-native Capabilities: Multi-agent Collaboration

|  Short Name  |   Paper   |   Date  |  Code/Project Link |
|  :---------: |   :---:   |   :--:  |  :---------------: |
|   CAMEL  |  [CAMEL: Communicative Agents for “Mind” Exploration of Large Language Model Society](https://arxiv.org/pdf/2303.17760)  |  2023-03  | [GitHub](https://github.com/camel-ai/camel) ![Stars](https://img.shields.io/github/stars/camel-ai/camel) [Project](https://www.camel-ai.org/) |
|   MetaGPT  |  [MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](https://arxiv.org/pdf/2308.00352)  |  2023-08  | [GitHub](https://github.com/FoundationAgents/MetaGPT) ![Stars](https://img.shields.io/github/stars/FoundationAgents/MetaGPT) |
|   MAD  |  [Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate](https://arxiv.org/abs/2305.19118)  |  2023-05  | [GitHub](https://github.com/Skytliang/Multi-Agents-Debate) ![Stars](https://img.shields.io/github/stars/Skytliang/Multi-Agents-Debate) |
|   MoA  |  [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692)  |  2024-06  | [GitHub](https://github.com/togethercomputer/moa) ![Stars](https://img.shields.io/github/stars/togethercomputer/moa) |
|   AFlow  |  [AFlow: Automating Agentic Workflow Generation](https://arxiv.org/abs/2410.10762)  |  2024-10  | [GitHub](https://github.com/FoundationAgents/AFlow) ![Stars](https://img.shields.io/github/stars/FoundationAgents/AFlow) |
|   MALT  |  [MALT: Improving Reasoning with Multi-Agent LLM Training](https://arxiv.org/abs/2412.01928)  |  2024-12  | - |
|   CORY  |  [Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2410.06101)  |  2024-10  | [GitHub](https://github.com/Harry67Hu/CORY) ![Stars](https://img.shields.io/github/stars/Harry67Hu/CORY) |
|   MARFT  |  [MARFT: Multi-Agent Reinforcement Fine-Tuning](https://arxiv.org/abs/2504.16129v2)  |  2025-04  | [GitHub](https://github.com/jwliao-ai/MARFT) ![Stars](https://img.shields.io/github/stars/jwliao-ai/MARFT) |
|   MAGRPO  |  [LLM Collaboration With Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2508.04652)  |  2025-08  | - |
|   RLCCF  |  [Wisdom of the Crowd: Reinforcement Learning from Coevolutionary Collective Feedback](https://arxiv.org/abs/2508.12338)  |  2025-08  | - |
|   MATPO  |  [Multi-Agent Tool-Integrated Policy Optimization](https://arxiv.org/abs/2510.04678)  |  2025-10  | [GitHub](https://github.com/mzf666/MATPO) ![Stars](https://img.shields.io/github/stars/mzf666/MATPO) |
|   MasHost  |  [MasHost Builds It All: Autonomous Multi-Agent System Directed by Reinforcement Learning](https://arxiv.org/pdf/2506.08507)  |  2025-06  | - |
|   G-Designer  |  [G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks](https://arxiv.org/pdf/2410.11782)  |  2024-10  | [GitHub](https://github.com/yanweiyue/GDesigner) ![Stars](https://img.shields.io/github/stars/yanweiyue/GDesigner) |
|   ARG-Designer  |  [Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation](https://arxiv.org/pdf/2507.18224v1)  |  2025-07  | [GitHub](https://github.com/Shiy-Li/ARG-Designer) ![Stars](https://img.shields.io/github/stars/Shiy-Li/ARG-Designer) |
|   MAGDi  |  [MAGDi: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models](https://github.com/dinobby/MAGDi)  |  2024-02  | [GitHub](https://github.com/dinobby/MAGDi) ![Stars](https://img.shields.io/github/stars/dinobby/MAGDi) |
|   CoA  |  [Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL](https://arxiv.org/pdf/2508.13167)  |  2025-08  | [GitHub](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models) ![Stars](https://img.shields.io/github/stars/OPPO-PersonalAI/Agent_Foundation_Models) |

### 7.1.2&nbsp;&nbsp;Emerging Model-native Capabilities: Pipeline-based paradigm

|  Short Name  |   Paper   |   Date  |  Code/Project Link |
|  :---------: |   :---:   |   :--:  |  :---------------: |
|   Reflexion  |  [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366)  |  2023-03  | [GitHub](https://github.com/noahshinn/reflexion) ![Stars](https://img.shields.io/github/stars/noahshinn/reflexion) |
|   Self-Refine  |  [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)  |  2023-03  | [GitHub](https://github.com/madaan/self-refine) ![Stars](https://img.shields.io/github/stars/madaan/self-refine) [Project](https://selfrefine.info/) |
|   Critic  |  [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://arxiv.org/abs/2305.11738)  |  2023-05  | [GitHub](https://github.com/microsoft/ProphetNet/tree/master/CRITIC) ![Stars](https://img.shields.io/github/stars/microsoft/ProphetNet) |
|   Conf. Matters  |  [Confidence Matters: Revisiting Intrinsic Self-Correction Capabilities of Large Language Models](https://arxiv.org/abs/2402.12563)  |  2024-02  | [GitHub](https://github.com/MBZUAI-CLeaR/IoE-Prompting) ![Stars](https://img.shields.io/github/stars/MBZUAI-CLeaR/IoE-Prompting) |
|   SCoRe  |  [Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917)  |  2024-09  | - |
|   CoV  |  [Retrieving, Rethinking and Revising: The Chain-of-Verification Can Improve Retrieval Augmented Generation](https://arxiv.org/abs/2410.05801)  |  2024-10  | - |
|   DPSDP  |  [Reinforce LLM Reasoning through Multi-Agent Reflection](https://arxiv.org/pdf/2506.08379)  |  2025-06  | - |
|   Agent-R  |  [Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training](https://arxiv.org/pdf/2501.11425)  |  2025-01  | [GitHub](https://github.com/bytedance/Agent-R) ![Stars](https://img.shields.io/github/stars/bytedance/Agent-R) |
|   AgentRefine  |  [AgentRefine: Enhancing Agent Generalization through Refinement Tuning](https://arxiv.org/pdf/2501.01702.pdf)  |  2025-01  | [GitHub](https://github.com/Fu-Dayuan/AgentRefine) ![Stars](https://img.shields.io/github/stars/Fu-Dayuan/AgentRefine) |
|   STeP  |  [Training LLM-Based Agents with Synthetic Self-Reflected Trajectories and Partial Masking](https://arxiv.org/pdf/2505.20023)  |  2025-05  | - |
|   KnowSelf  |  [Agentic Knowledgeable Self-awareness](https://arxiv.org/abs/2504.03553)  |  2025-04  | [GitHub](https://github.com/zjunlp/KnowSelf) ![Stars](https://img.shields.io/github/stars/zjunlp/KnowSelf) |
