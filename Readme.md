# Repository of CasualBench



### Overview

Implementation for paper CausalBench: A Comprehensive Benchmark for Causal Learning Capability of LLMs


<img src="figs/framework.png" style="width: auto; height: auto;" alt="framework"> 



### Description of CausalBench

The ability to understand causality significantly impacts the competence of large language models (LLMs) in output explanation and counterfactual reasoning, as causality reveals the underlying data distribution. However, the lack of a comprehensive benchmark currently limits the evaluation of LLMs' causal learning capabilities. To fill this gap, CausalBench is developed based on data from the causal research community, enabling comparative evaluations of LLMs against traditional causal learning algorithms. To provide a comprehensive investigation, we offer three tasks of varying difficulties, including correlation, causal skeleton, and causality identification. Evaluations of 19 leading LLMs reveal that, while closed-source LLMs show potential for simple causal relationships, they significantly lag behind traditional algorithms on larger-scale networks (> 50 nodes). Specifically, LLMs struggle with collider structures but excel at chain structures, especially at long-chain causality analogous to Chains-of-Thought techniques. This supports the current prompt approaches while suggesting directions to enhance LLMs' causal reasoning capability. Furthermore, CausalBench incorporates background knowledge and training data into prompts to thoroughly unlock LLMs' text comprehension ability during evaluation, whose findings indicate that LLM understand causality through semantic associations with distinct entities, rather than directly from contextual information or numerical distributions.

CausalBench collects data from Bnlearn. After that, we extract variable names from the collected datasets and design four types of prompt formats. Then, CausalBench creates three core evaluation tasks, including correlation, causal
skeleton, and causality identification tasks. In order to gain deeper insights into the overall causal learning capabilities of LLMs, we evaluate three closed-source LLMs, i.e. GPT3.5-Turbo, GPT4, and GPT4-Turbo, along with five series
of open-source LLMs: BERT series, LLAMA series (i.e., 7B, 13B and 33B), OPT series [30] (i.e., 1.3B, 2.7B, 6.7B and 66B), Falcon series (i.e., 7B and 40B), and InternLM series (i.e., 7B and 20B). As shown in the Figure, the resultant CausalBench possesses at least four advantages:

- Diverse scales of datasets from the causal learning community: CausalBench is an extension of the causal learning research community’s efforts, designed to offer a robust evaluation framework. It incorporates 15 commonly used real-world causal learning datasets of diverse scales, enabling rigorous and quantitative measurement of LLMs' causal learning capacities with extensive evaluation results in the causal research community as a reference.

- Evaluation tasks of varying depths and difficulties: CausalBench offers three tasks of different difficulties, i.e., correlation, causal skeleton, and causality identification, respectively, to holistically assess the causal learning capabilities of existing LLMs. Additionally, we have provided an example of causal structures similar to a particular prompt technique, which identifies the long cause-effect chain to evaluate causal learning abilities on multi-step reasoning like the popular Chain-of-Thought (CoT) prompt technique.


- Diverse prompts with rich information: CausalBench offers four distinct prompt formats, encompassing variable names and their combinations with background knowledge and training data, respectively, and the combination of the three. With these diverse prompts, CausalBench can well assess LLMs' causal learning capacities through looking into their abilities to utilize prior information and comprehend long-text in understanding causal relations.

- Demonstration of the upper limit of LLMs' causal learning capability across various scales and complexities: CausalBench evaluates causal relations of varying scales and complexities. CausalBench covers causal learning datasets with scales ranging from 5 to 109 nodes, far exceeding what current evaluation works have explored. Meanwhile, it evaluates various types of causal structures and discusses different densities in causal learning networks.	

- All the instances were released after the paper review process.

- Please refer to the article [url] for more details.



### Repository Content

This repository contains the implementation for benchmark construction and evaluation.

- `benchmark` directory contains detailed inference codes of three core evaluation tasks, including correlation, causal skeleton, and causality identification tasks, and four types of prompt formats (variable name, variable name + background knowledge, variable name + structured data, variable name + background knowledge + structured data) to evaluate LLMs' causal learning capabilities. 
- `dataset_construction` directory contains the different prompts of three core evaluation tasks, four types of prompt formats (variable name, variable name + background knowledge, variable name + structured data, variable name + background knowledge + structured data), and other evaluation analysis about data identification, causal network, and prompt robustness
- `evaluation` directory contains the implementation for evaluating the causal learning capabilities of LLMs with different LLMs based on the labels and inference results.
- `causal_learning_data` includes the raw data in the causal learning community.
-  More elements will be added to the repository soon (Current repo gives an instance (i.e., asia dataset) for reference).



### Citations

Please cite the paper and star the repo if you use CausalBecnh and find it helpful. Feel free to contact zy-yu.zhou@connect.polyu.hk if you have any questions.

```latex
@misc{zhou2024causalbench,
      title={CausalBench: A Comprehensive Benchmark for Causal Learning Capability of Large Language Models}, 
      author={Zhou, Yu and Wu, Xingyu and Huang, Beicheng and Wu, Jibin and Feng, Liang and Tan, Kay Chen},
      year={2024},
      eprint={2404.06349},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```
