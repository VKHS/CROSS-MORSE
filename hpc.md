# HPC and local paths

Public source code must not contain a maintainer's account name, allocation, login host, home directory, scratch path, or output-copy command.

Use environment variables instead:

```bash
set -a
source release/hpc/cluster.env
set +a
```

Copy `release/hpc/cluster.env.example` to an untracked `release/hpc/cluster.env` and fill it locally.

Portable Slurm directives should write logs relative to the submission directory, for example:

```bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
```

Do not commit `scp` commands containing personal usernames or hostnames. Document result collection generically in dataset-specific READMEs.
