# Dataset evidence and usage policy

## Selected public dataset: Loghub HDFS v1

NeuralOps selects **HDFS v1** from the official LogPAI Loghub collection for its public evaluation track.

| Field | Verified value |
| --- | --- |
| Record | Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics |
| Maintainer | Curated by LOGPAI |
| DOI | [10.5281/zenodo.8196385](https://doi.org/10.5281/zenodo.8196385) |
| Dataset license | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| HDFS v1 archive | `HDFS_v1.zip` |
| Archive size | `186645559` bytes |
| Archive checksum | `md5:76a24b4d9a6164d543fb275f89773260` |
| Dataset scale | 11,175,629 log lines; 1.47 GiB uncompressed, per Loghub |
| Unit of prediction | HDFS block/session sequence |
| Public target | binary normal/anomaly |

Verified on 2026-08-05 from the [official Zenodo record](https://zenodo.org/records/8196385) and [official Loghub repository](https://github.com/logpai/loghub/tree/master/HDFS).

### License boundary

The NeuralOps code is MIT licensed. Loghub data is a separate CC BY 4.0 work. NeuralOps does not commit or relicense the dataset. The downloader obtains the exact official archive, verifies its checksum, and preserves the upstream attribution notice. Anyone distributing copies must keep the Loghub attribution and cite the paper where applicable.

### Required citation

Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, and Michael R. Lyu. *Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics*. ISSRE 2023.

The original HDFS anomaly-detection dataset originates from Wei Xu, Ling Huang, Armando Fox, David Patterson, and Michael Jordan. *Detecting Large-Scale System Problems by Mining Console Logs*. SOSP 2009.

## Task formulation

Each example is the ordered event-template sequence associated with one HDFS block ID. The target is exactly the upstream block-level normal/anomaly label. HDFS v1 does not provide trustworthy category or severity targets; both outputs are disabled for this track.

The full public benchmark is the scientific target. CI uses a tiny, clearly named generated fixture solely to verify code paths. Fixture metrics are never presented as benchmark results.

## OpsForge Sim v1

OpsForge Sim v1 is a NeuralOps-owned synthetic generator used for product behavior and multi-task experiments. It generates variable parameters and event order from incident scenarios without inserting label words into event tokens. It supports:

- binary anomaly status;
- nine synthetic incident categories;
- low, medium, or high synthetic severity.

All reports, API metadata, and UI screens identify this source as synthetic. Its results test implementation behavior, not generalization to real systems.

## Alternatives considered

- **BGL:** useful line-level anomaly labels, but less aligned with the requested session-sequence product surface.
- **Hadoop:** scenario-level labels and fewer lines, but HDFS has a better established block-sequence formulation and baseline ecosystem.
- **Private OpsForge logs:** rejected as the sole research dataset because private synthetic labels cannot substantiate a public benchmark claim.

