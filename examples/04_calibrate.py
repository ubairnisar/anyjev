"""Pick the threshold that keeps recall on structured crops at 100%.

labels.csv:
    path,label
    crops/001.png,structured
    crops/002.png,prose
"""
import csv

from anyjev import Judge, presets

task = presets.structured_vs_prose()
rows = list(csv.DictReader(open("labels.csv")))

cal = Judge().calibrate(task, [r["path"] for r in rows], [r["label"] for r in rows], target_recall=1.0)
print(cal)
task.threshold = cal.threshold
task.save("structured-vs-prose.yaml")
