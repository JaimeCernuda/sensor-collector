# ChronoTick Paper — Project Instructions

## Page Budget
- **10 pages of content** is the limit. Citations overflow beyond page 10 and do not count.
- Citations are currently ~1.25 pages.
- When evaluating page count from the compiled PDF, subtract citation pages to get content pages.
- Target venue: IEEE conference (IEEEtran two-column format).

## File Structure
- The REVISED section files use numbered prefixes: `sections/1.a.introduction.tex`, `sections/2.related-work.tex`, `sections/3.design.tex`, `sections/4.evaluations.tex`, `sections/5.conclusions.tex`
- The unnumbered files (`sections/introduction.tex`, etc.) are OLD stale versions — do not use.
- `main.tex` references the revised files.

## External Data
- Multivariate sensor evaluation results are in `D:\Libraries\Documents\projects\sensor_gathering\paper`
- These contain multi-model results with 8-hour and 16-hour checkpoints (24-hour run still in progress)
- This data supports Contribution C3 (multivariate compensation from commodity sensors)

## Writing Style
- Do NOT use em-dashes (---) in LaTeX prose. Use commas, semicolons, "such as", or restructure the sentence instead. Em-dashes read as AI-generated writing.
