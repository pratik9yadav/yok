"""The repo's pytest run must not collect the exam tasks' own files: their hidden tests import solution modules that
only exist inside a grading workspace (coder/exam/grade.py runs them there)."""

collect_ignore = ["tasks", "sample_tasks"]
