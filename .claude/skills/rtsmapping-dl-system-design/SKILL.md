---
name: rtsmapping-dl-system-design
description: Deep learning system design best practices for rtsmapping DL model development, including system design and code implementation. Covers the full lifecycle including data, training, inference and post-inference. The design and code implementation details should follow the markdown files in their own directories.
---

# Overall Guidance

---
- This document serves as a high-level guide to ensure consistency, quality, and reproducibility across all aspects of the project. 

- The full project lifecyle composes of four main components: data, training, inference and post-inference. The development of each model iteration and component should follow the markdown files in their own directories (e.g., `data/data.md`, `model/model.md`, `training/training.md`). There are also guidelines for computing and dockerization in `computing/` and `docker_training/`, the design of relevant parts should follow the markdown files in those directories. 

- The development of the system should follow the best practices in AI and software engineering, but also don't overkill with overly complex design and engineering, remember this is a solo research project, the design should be simple and efficient, but also maintainable and reproducible. 

- the environment should be dockerised

- the experiments should be tracked with MLflow

- the process and results should be documented in details and in a living document (e.g., a markdown file in `docs/`), the documentation should be updated with every iteration and should include the design decisions, the implementation details, the results and the analysis.

- the training and inference should be consistent in terms of data processing 

- input, output and intermediate data should follow the data format standards in `data/data_format.md`
