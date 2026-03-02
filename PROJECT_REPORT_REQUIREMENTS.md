# Project Report Website (GitHub Pages)

Your final deliverable is a public-facing project report written as a Jupyter notebook, converted to HTML, and hosted at a stable URL via GitHub Pages. Think of it as a short applied machine learning paper written for an educated social science audience.

This website is also a portfolio piece: you will be able to share it with future employers, graduate programs, or include it on your CV as evidence of your ability to design, implement, evaluate, and communicate a machine learning project end to end.

## Requirements

### 1) A Public URL (GitHub Pages)
- A working GitHub Pages link (e.g., https://yourname.github.io/your-repo/)
- The page must load without logging in and display your report clearly
- The URL must remain stable through the end of the term

### 2) A Notebook-Based Report Converted to HTML
Your report must be written as a Jupyter notebook (.ipynb) and published as an HTML page

The HTML must include:
- Text, figures, and tables
- Key results (metrics + interpretation)
- Enough context that a reader can follow the project without opening the raw notebook

### 3) Your GitHub Repository
In addition to all of your previous work, repository should contain:
- The source notebook (.ipynb)
- The rendered HTML and supporting files (in a separate folder to your previous work)
- Everything you've done in the previous weeks
- Clear instructions in the README for reproduction

## Suggested Sections in the Final Report

Your report should include the following sections (use headings):

### Title + Authors
- Project title
- Group member names

### Research Question and Motivation
- What is the social science question or problem?
- Why does it matter?
- What would success look like?

### Data
- Data source(s) and provenance
- Unit of analysis
- Key variables / labels
- Any important preprocessing choices
- Ethical considerations (privacy, bias, sensitive content) where relevant

### Problem Setup
- What is the ML task (classification/regression/etc.)?
- What is the target variable?
- What are the inputs/features?
- What assumptions are you making?

### Methods
- Baseline(s)
- Final model(s) and why they were chosen
- Feature engineering / representations
- Training and tuning approach (as relevant)

### Evaluation
- Train/test split strategy (and why it is appropriate)
- Metrics (and why those metrics match the goal)
- Main quantitative results (include at least one clear table or figure)

### Error Analysis / Diagnostics
- Where does the model fail and why?
- At least one concrete diagnostic (confusion pattern, residual analysis, qualitative inspection, etc.)
- At least one robustness/sensitivity check where appropriate

### Interpretation and Substantive Takeaways
- What do the results imply for the social science question?
- What should a reader conclude—and what should they not conclude?

### Limitations and Future Work
- Data limitations, measurement issues, generalizability
- What you would do next with more time or better data

### References
- Cite any papers, datasets, and tools you relied on
- If you used generative AI, include an acknowledgment

## Minimum Content Expectations

- **Length**: ~1,500–2,000 words
- **Figures/Tables**: At least 3 that support your argument (not decorative)
- **Results**: At least one baseline and one improved approach (unless justified)
- **Clarity**: A reader should understand the full pipeline and the main conclusions from the HTML alone

## Website and Formatting Requirements

The GitHub Pages site must:
- Render cleanly in a browser (no broken images, missing equations, or huge unreadable cells)
- Have readable headings and a clear narrative flow
- Include your GitHub repo link prominently near the top
- Your final page should not require the reader to run code to understand your results

## Reproducibility Requirements (Repository)

Your repo must include:

**README.md** with:
- Project summary
- Link to the GitHub Pages report
- Instructions to reproduce results (environment + run order)
- A list of dependencies (requirements.txt or equivalent)
- A clear structure (e.g., data/, notebooks/, src/, figures/)
- A way to obtain data (either included if allowed, or instructions/scripts)

If your data cannot be shared publicly, your report must still describe how it was obtained and how it was processed, and your repo should include everything else necessary to reproduce given access.
