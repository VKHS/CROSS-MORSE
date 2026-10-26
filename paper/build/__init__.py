"""Run-driven paper build: evidence extraction, tables and figures from sealed run directories.

Everything in this package reads completed run directories (``examples/runs/<stem>`` or
``runs/<stem>``) and writes tables and figures from them.  Nothing here reads the
manuscript registry in ``paper/reported_results.json``; that registry is used only by the
separate comparison tools.  NumPy is required; Matplotlib only for figures.
"""
