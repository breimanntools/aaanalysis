# AAanalysis – Amino acid analysis

AAanalysis is a platform to analyse amino acid scales and to perform comparative physicochemical
profiling of protein sequences. The central tool of AAanalysis is a rigorous clustering algorithm
called AAclust which was developed (a) to classify 586 amino acid scales into 8 categories and 86 subcategories
and (b) to select a redundancy-reduced subset of amino acid scales.

This subset of scales can then be used for sequence-based machine learning pipelines.


## Use the web server

The webserver is available under ...

## To try it local

Clone the repository 

```
git clone https://github.com/stephanbreimann/AAanalysis
```

Set up a virtual environment and install dependencies using the requirements.txt file

```
pip install -r requirements.txt
```

Run dash app to explore the amino acid scales and their relations on the level categories,
subcategories, scales, and amino acids

```
python app_dashboard.py
```

## Explore the amino acid scale classification
586 amino acid scales (from AAindex; Lins et al., 2003; and Koehler et al., 2009) were classified into
8 categories and 86 subcategories using AAclust.

You can find the 586 numerical scales in ```data/scales.xlsx``` and the scale classification in 
```data/scale_classification.xlsx```. The unique identifier of the amino acid scales was taken from AAindex
and adapted for scales from Lins et al. and Koehler et al., accordingly.



