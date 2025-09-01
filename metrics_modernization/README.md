Selected Datasets:

1. AmnestyQA: https://huggingface.co/datasets/explodinggradients/amnesty_qa
2. FiQA: https://huggingface.co/datasets/explodinggradients/fiqa

Selected Metrics:

1. faithfulness
2. answer_relevancy
3. answer_correctness
4. context_recall
5. context_precision

For each metric, we will need the following comparisons:

| Metric             | Dataset   | Current | Modern w/ Direct Prompt | Modern w/ 2 Step + Identical Prompts |
| ------------------ | --------- | ------- | ----------------------- | ------------------------------------ |
| faithfulness       | AmnestyQA | 0.6092  | -                       | -                                    |
| answer_relevancy   | AmnestyQA | 0.7831  | -                       | -                                    |
| answer_correctness | AmnestyQA | 0.6092  | -                       | -                                    |
| context_recall     | AmnestyQA | 0.7831  | -                       | -                                    |
| context_precision  | AmnestyQA | 0.6092  | -                       | -                                    |
