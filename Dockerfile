FROM continuumio/miniconda3

WORKDIR /

# copy environment file first (better caching)
COPY environment.yml .

# install dependencies
RUN conda env create -f environment.yml && conda clean -afy

# set shell for subsequent RUNs
SHELL ["conda", "run", "-n", "transformers", "/bin/bash", "-c"]

# copy all Python files in project root (same folder as Dockerfile/Makefile)
COPY ./*.py ./

# default command
CMD ["conda", "run", "--no-capture-output", "-n", "transformers", "python", "-u", "main.py"]