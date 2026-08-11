# Tabular datasets

The dataset files are not redistributed. Obtain the original multi-label
datasets under their respective terms and place matching `.arff` and `.xml`
files here:

```text
bibtex.arff
bibtex.xml
enron.arff
enron.xml
slashdot.arff
slashdot.xml
yahoo-arts.arff
yahoo-arts.xml
yahoo-education.arff
yahoo-education.xml
yahoo-recreation.arff
yahoo-recreation.xml
```

The XML files define label attributes. The ARFF files contain features and
labels. Do not alter the raw files to improve metrics.

Set `CREM_DATA_DIR` if you keep these files outside the repository. Protocol-v2
caches are generated automatically in `cache/` and are ignored by Git.

